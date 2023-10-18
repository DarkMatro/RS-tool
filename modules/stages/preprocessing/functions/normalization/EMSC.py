import numpy as np
import scipy.optimize
from numba import njit
from sklearn.decomposition import _incremental_pca


def konevskikh_parameters(a, n0, f):
    """
    Compute parameters for Konevskikh algorithm
    :param a: cell radius
    :param n0: refractive index
    :param f: scaling factor
    :return: parameters alpha0 and gamma
    """
    alpha0 = 4.0 * np.pi * a * (n0 - 1.0)
    gamma = np.divide(f, n0 - 1.0)
    return alpha0, gamma


def GramSchmidt(V):
    """
    Perform Gram-Schmidt normalization for the matrix V
    :param V: matrix
    :return: nGram-Schmidt normalized matrix
    """
    V = np.array(V)
    U = np.zeros(np.shape(V))

    for k in range(len(V)):
        sum1 = 0
        for j in range(k):
            sum1 += np.dot(V[k], U[j]) / np.dot(U[j], U[j]) * U[j]
        U[k] = V[k] - sum1
    return U


def check_orthogonality(U):
    """
    Check orthogonality of a matrix
    :param U: matrix
    """
    for i in range(len(U)):
        for j in range(i, len(U)):
            if i != j:
                print(np.dot(U[i], U[j]))


def find_nearest_number_index(array, value):
    """
    Find the nearest number in an array and return its index
    :param array:
    :param value: value to be found inside the array
    :return: position of the number closest to value in array
    """
    array = np.array(array)  # Convert to numpy array
    if np.shape(np.array(value)) == ():  # If only one value wants to be found:
        index = (np.abs(array - value)).argmin()  # Get the index of item closest to the value
    else:  # If value is a list:
        value = np.array(value)
        index = np.zeros(np.shape(value))
        k = 0
        # Find the indexes for all values in value
        for val in value:
            index[k] = (np.abs(array - val)).argmin()
            k += 1
        index = index.astype(int)  # Convert the indexes to integers
    return index


@njit(fastmath=True)
def Q_ext_kohler(wn, alpha):
    """
    Compute the scattering extinction values for a given alpha and a range of wavenumbers
    :param wn: array of wavenumbers
    :param alpha: scalar alpha
    :return: array of scattering extinctions calculated for alpha in the given wavenumbers
    """
    rho = alpha * wn
    Q = 2.0 - (4.0 / rho) * np.sin(rho) + (2.0 / rho) ** 2.0 * (1.0 - np.cos(rho))
    return Q


def apparent_spectrum_fit_function(wn, Z_ref, p, b, c, g):
    """
    Function used to fit the apparent spectrum
    :param wn: wavenumbers
    :param Z_ref: reference spectrum
    :param p: principal components of the extinction matrix
    :param b: Reference's linear factor
    :param c: Offset
    :param g: Extinction matrix's PCA scores (to be fitted)
    :return: fitting of the apparent specrum
    """
    A = b * Z_ref + c + np.dot(g, p)  # Extended multiplicative scattering correction formula
    return A


def reference_spectrum_fit_function(wn, p, c, g):
    """
    Function used to fit a reference spectrum (without using another spectrum as reference).
    :param wn: wavenumbers
    :param p: principal components of the extinction matrix
    :param c: offset
    :param g: PCA scores (to be fitted)
    :return: fitting of the reference spectrum
    """
    A = c + np.dot(g, p)
    return A


def apparent_spectrum_fit_function_Bassan(wn, Z_ref, p, c, m, h, g):
    """
    Function used to fit the apparent spectrum in Bassan's algorithm
    :param wn: wave numbers
    :param Z_ref: reference spectrum
    :param p: principal components of the extinction matrix
    :param c: offset
    :param m: linear baseline
    :param h: reference's linear factor
    :param g: PCA scores to be fit
    :return: fitting of the apparent spectrum
    """
    A = c + m * wn + h * Z_ref + np.dot(g, p)
    return A


def correct_reference(m, wn, a, d, w_regions):
    """
    Correct reference spectrum as in Kohler's method
    :param m: reference spectrum
    :param wn: wavenumbers
    :param a: Average refractive index range
    :param d: Cell diameter range
    :param w_regions: Weighted regions
    :return: corrected reference spectrum
    """
    n_components = 6  # Set the number of principal components

    # Copy the input variables
    m = np.copy(m)
    wn = np.copy(wn)

    # Compute the alpha range:
    alpha = 4.0 * np.pi * 0.5 * np.linspace(np.min(d) * (np.min(a) - 1.0), np.max(d) * (np.max(a) - 1.0), 150)

    p0 = np.ones(1 + n_components)  # Initial guess for the fitting

    # Compute extinction matrix
    Q_ext = np.zeros((np.size(alpha), np.size(wn)))
    for i in range(np.size(alpha)):
        Q_ext[i][:] = Q_ext_kohler(wn, alpha=alpha[i])

    # Perform PCA to Q_ext
    pca = _incremental_pca.IncrementalPCA(n_components=n_components)
    pca.fit(Q_ext)
    p_i = pca.components_  # Get the principal components of the extinction matrix

    # Get the weighted regions of the wave numbers, the reference spectrum and the principal components
    w_indexes = []
    for pair in w_regions:
        min_pair = min(pair)
        max_pair = max(pair)
        ii1 = find_nearest_number_index(wn, min_pair)
        ii2 = find_nearest_number_index(wn, max_pair)
        w_indexes.extend(np.arange(ii1, ii2))
    wn_w = np.copy(wn[w_indexes])
    m_w = np.copy(m[w_indexes])
    p_i_w = np.copy(p_i[:, w_indexes])

    def min_fun(x):
        """
        Function to be minimized for the fitting
        :param x: offset and PCA scores
        :return: difference between the spectrum and its fitting
        """
        cc, g = x[0], x[1:]
        # Return the squared norm of the difference between the reference spectrum and its fitting:
        return np.linalg.norm(m_w - reference_spectrum_fit_function(wn_w, p_i_w, cc, g)) ** 2.0

    # Perform the minimization using Powell method
    res = scipy.optimize.minimize(min_fun, p0, bounds=None, method='Powell')

    c, g_i = res.x[0], res.x[1:]  # Obtain the fitted parameters

    # Apply the correction:
    m_corr = np.zeros(np.shape(m))
    for i in range(len(wn)):
        sum1 = 0
        for j in range(len(g_i)):
            sum1 += g_i[j] * p_i[j][i]
        m_corr[i] = (m[i] - c - sum1)

    return m_corr  # Return the corrected spectrum


def Kohler(x_axis: np.ndarray, y_axis: np.ndarray, ref_array: np.ndarray, n_components: int = 8) -> np.ndarray:
    """
    Correct scattered spectra using Kohler's algorithm
    :param x_axis: array of wavenumbers
    :param y_axis: apparent spectrum
    :param ref_array: reference spectrum
    :param n_components: number of principal components to be calculated 
    :return: corrected data
    """
    # Initialize the alpha parameter:
    alpha = np.linspace(3.14, 49.95, 150) * 1.0e-4  # alpha = 2 * pi * d * (n - 1) * wavenumber
    p0 = np.ones(2 + n_components)  # Initialize the initial guess for the fitting

    # # Initialize the extinction matrix:
    Q_ext = np.zeros((np.size(alpha), np.size(x_axis)))
    for i in range(np.size(alpha)):
        Q_ext[i][:] = Q_ext_kohler(x_axis, alpha=alpha[i])

    # Perform PCA of Q_ext:
    pca = _incremental_pca.IncrementalPCA(n_components=n_components)
    pca.fit(Q_ext)
    p_i = pca.components_  # Extract the principal components

    # print(np.sum(pca.explained_variance_ratio_)*100)  # Print th explained variance ratio in percentage

    def min_fun(x):
        """
        Function to be minimized by the fitting
        :param x: array containing the reference linear factor, the offset, and the PCA scores 
        :return: function to be minimized
        """
        bb, cc, g = x[0], x[1], x[2:]
        # Return the squared norm of the difference between the apparent spectrum and the fit
        return np.linalg.norm(y_axis - apparent_spectrum_fit_function(x_axis, ref_array, p_i, bb, cc, g)) ** 2.0

    # Minimize the function using Powell method
    res = scipy.optimize.minimize(min_fun, p0, method='Powell')
    # print(res)  # Print the minimization result
    # assert(res.success) # Raise AssertionError if res.success == False

    b, c, g_i = res.x[0], res.x[1], res.x[2:]  # Obtain the fitted parameters

    # Apply the correction to the apparent spectrum
    z_corr = np.zeros(np.shape(ref_array))
    for i in range(len(x_axis)):
        sum1 = 0
        for j in range(len(g_i)):
            sum1 += g_i[j] * p_i[j][i]
        z_corr[i] = (y_axis[i] - c - sum1) / b

    return z_corr  # Return the correction in reverse order for compatibility
