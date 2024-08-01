# pylint: disable=no-name-in-module, too-many-lines, invalid-name, import-error
"""
This module provides a set of utility functions for various tasks in spectral data processing
and machine learning model fitting. It includes methods for baseline correction, normalization,
smoothing, peak shape parameterization, and machine learning classifiers. The module is designed
to facilitate the processing and analysis of spectral data using different baseline correction
techniques, normalization methods, and smoothing algorithms. Additionally, it provides a means
to configure and fit machine learning models for classification tasks.

Functions:
    - baseline_parameter_defaults: Returns default parameters for various baseline correction
        methods.
    - baseline_methods: Returns a dictionary of baseline correction methods with their corresponding
        functions and sample limits.
    - classificator_funcs: Returns a dictionary of machine learning classifier fitting functions.
    - objectives: Returns a dictionary of objective functions for hyperparameter optimization.
    - normalize_methods: Returns a dictionary of normalization methods with their corresponding
        functions and sample limits.
    - peak_shapes_params: Returns a dictionary of peak shape functions and additional parameters.
    - smoothing_methods: Returns a dictionary of smoothing methods with their corresponding
        functions and sample limits.
    - get_optuna_params: Returns a dictionary of hyperparameters for Optuna optimization for various
        classifiers.
"""

from typing import Callable

from src.stages.fitting.functions.peak_shapes import gaussian, split_gaussian, skewed_gaussian, \
    lorentzian, split_lorentzian, \
    voigt, split_voigt, skewed_voigt, pseudovoigt, split_pseudovoigt, pearson4, split_pearson4, \
    pearson7, \
    split_pearson7
from src.stages.ml.functions.fit_classificators import (
    fit_lda, fit_lr, fit_svc, fit_dt, fit_rf, fit_xgboost)
from src.stages.ml.functions.hyperopt import objective_lda, objective_lr, objective_svc, \
    objective_dt, objective_rf
from src.stages.preprocessing.functions.baseline_correction import baseline_correct, ex_mod_poly, \
    baseline_asls, baseline_arpls, \
    baseline_airpls, baseline_drpls
from src.stages.preprocessing.functions.normalization.normalization import normalize_emsc, \
    normalize_snv, normalize_area, normalize_trapz_area, \
    normalize_max, normalize_minmax
from src.stages.preprocessing.functions.smoothing.functions_smoothing import smooth_mlesg, \
    smooth_ceemdan, smooth_eemd, smooth_emd, smooth_savgol, whittaker, \
    smooth_flat, smooth_window, smooth_window_kaiser, smooth_med_filt, smooth_wiener


def baseline_parameter_defaults() -> dict[str, dict[str, int | float]]:
    """
    Returns default parameters for various baseline correction methods.

    Returns
    -------
    dict
        A dictionary where keys are baseline correction method names and values
        are dictionaries containing parameter names and their default values.
    """
    return {'Poly': {'polynome_degree': 5},
            'ModPoly': {'polynome_degree': 5, 'tol': 1e-3, 'max_iter': 250},
            'iModPoly': {'polynome_degree': 6, 'tol': 1e-3, 'max_iter': 250, 'num_std': 0.},
            'ExModPoly': {'polynome_degree': 7, 'tol': 1e-6, 'max_iter': 100, 'quantile': 1e-5,
                          'scale': .5, 'num_std': 3.,
                          'half_window': 2},
            'Penalized poly': {'polynome_degree': 2, 'tol': 1e-3, 'max_iter': 250,
                               'alpha_factor': 0.99999},
            'LOESS': {'polynome_degree': 1, 'tol': 1e-3, 'max_iter': 10, 'fraction': 0.2,
                      'scale': 3.0},
            'Quantile regression': {'polynome_degree': 2, 'tol': 1e-6, 'max_iter': 250,
                                    'quantile': 0.01},
            'Goldindec': {'polynome_degree': 2, 'tol': 1e-3, 'max_iter': 250, 'peak_ratio': 0.5,
                          'alpha_factor': 0.99},
            'AsLS': {'lambda': 1000000, 'p': 1e-3, 'max_iter': 50},
            'iAsLS': {'lambda': 1000000, 'p': 1e-2, 'max_iter': 50, 'tol': 1e-3},
            'arPLS': {'lambda': 100000, 'p': 1e-6, 'max_iter': 50},
            'airPLS': {'lambda': 1000000, 'p': 1e-6, 'max_iter': 50},
            'drPLS': {'lambda': 100000, 'p': 1e-6, 'max_iter': 50, 'eta': 0.5},
            'iarPLS': {'lambda': 100, 'tol': 1e-6, 'max_iter': 50},
            'asPLS': {'lambda': 100000, 'tol': 1e-6, 'max_iter': 100},
            'psaLSA': {'lambda': 100000, 'p': 0.5, 'max_iter': 50, 'tol': 1e-3},
            'DerPSALSA': {'lambda': 1000000, 'p': 0.01, 'max_iter': 50, 'tol': 1e-3},
            'MPLS': {'lambda': 1000000, 'p': 0.0, 'max_iter': 50, 'tol': 1e-3},
            'iMor': {'max_iter': 200, 'tol': 1e-3},
            'MorMol': {'max_iter': 250, 'tol': 1e-3},
            'AMorMol': {'max_iter': 200, 'tol': 1e-3},
            'MPSpline': {'lambda': 10000, 'p': 0.0, 'spline_degree': 3},
            'JBCD': {'max_iter': 20, 'tol': 1e-2, 'alpha_factor': 0.1},
            'Mixture Model': {'lambda': 100000, 'p': 1e-2, 'spline_degree': 3, 'max_iter': 50,
                              'tol': 1e-3},
            'IRSQR': {'lambda': 100, 'quantile': 0.01, 'spline_degree': 3, 'max_iter': 100,
                      'tol': 1e-6},
            'Corner-Cutting': {'max_iter': 100},
            'Noise Median': {'half_window': 5},
            'IPSA': {'max_iter': 500},
            'RIA': {'tol': 1e-2, 'max_iter': 500},
            'Dietrich': {'num_std': 3.0, 'polynome_degree': 5, 'max_iter': 50, 'tol': 1e-3,
                         'half_window': 5,
                         'min_length': 2},
            'Golotvin': {'num_std': 2.0, 'half_window': 5, 'min_length': 2, 'sections': 32},
            'Std Distribution': {'num_std': 1.1, 'half_window': 5},
            'FastChrom': {'half_window': 5, 'min_length': 2, 'max_iter': 100},
            'FABC': {'lambda': 1000000, 'num_std': 3.0, 'min_length': 2}
            }


def baseline_methods() -> dict[str, tuple[Callable, int]]:
    """
    Returns a dictionary of baseline correction methods with their corresponding functions and
    sample limits.

    Returns
    -------
    dict
        A dictionary where keys are baseline correction method names and values
        are tuples containing the baseline correction function and the sample limit.
    """
    return {
        # Polynomial methods
        'Poly': (baseline_correct, 10_000),
        'ModPoly': (baseline_correct, 1200),
        'iModPoly': (baseline_correct, 5000),
        'ExModPoly': (ex_mod_poly, 24),
        'Penalized poly': (baseline_correct, 10_000),
        'Quantile regression': (baseline_correct, 740),
        'Goldindec': (baseline_correct, 160),
        # Whittaker-smoothing-based methods
        'AsLS': (baseline_asls, 190),
        'iAsLS': (baseline_correct, 2700),
        'arPLS': (baseline_arpls, 175),
        'airPLS': (baseline_airpls, 100),
        'drPLS': (baseline_drpls, 270),
        'iarPLS': (baseline_correct, 830),
        'asPLS': (baseline_correct, 330),
        'psaLSA': (baseline_correct, 3800),
        'DerPSALSA': (baseline_correct, 1540),
        # Morphological methods
        'MPLS': (baseline_correct, 1000),
        'Morphological': (baseline_correct, 1000),
        'iMor': (baseline_correct, 175),
        'MorMol': (baseline_correct, 790),
        'AMorMol': (baseline_correct, 175),
        'Rolling Ball': (baseline_correct, 1000),
        'MWMV': (baseline_correct, 1000),
        'Top-hat': (baseline_correct, 1000),
        'MPSpline': (baseline_correct, 750),
        'JBCD': (baseline_correct, 810),
        # Spline methods
        'Mixture Model': (baseline_correct, 240),
        'IRSQR': (baseline_correct, 890),
        'Corner-Cutting': (baseline_correct, 4500),
        # Smoothing-based methods
        'Noise Median': (baseline_correct, 1230),
        'SNIP': (baseline_correct, 800),
        'SWiMA': (baseline_correct, 580),
        'IPSA': (baseline_correct, 740),
        'RIA': (baseline_correct, 560),
        # Baseline/Peak Classification methods
        'Dietrich': (baseline_correct, 740),
        'Golotvin': (baseline_correct, 1000),
        'Std Distribution': (baseline_correct, 1000),
        'FastChrom': (baseline_correct, 740),
        'FABC': (baseline_correct, 770),
        # Miscellaneous methods
        'BEaDS': (baseline_correct, 700)
    }


def classificator_funcs() -> dict[str, callable]:
    """
    Returns a dictionary of machine learning classifier fitting functions.

    Returns
    -------
    dict
        A dictionary where keys are classifier names and values are the corresponding fitting
        functions.
    """
    return {'LDA': fit_lda, 'Logistic regression': fit_lr, 'SVC': fit_svc,
            'Decision Tree': fit_dt, 'Random Forest': fit_rf, 'XGBoost': fit_xgboost}


def objectives() -> dict[str, callable]:
    """
    Returns a dictionary of objective functions for hyperparameter optimization.

    Returns
    -------
    dict
        A dictionary where keys are classifier names and values are the corresponding objective
        functions.
    """
    return {'LDA': objective_lda, 'Logistic regression': objective_lr, 'SVC': objective_svc,
            'Decision Tree': objective_dt, 'Random Forest': objective_rf, 'XGBoost': None}


def normalize_methods() -> dict[str, tuple]:
    """
    Returns a dictionary of normalization methods with their corresponding functions and sample
    limits.

    Returns
    -------
    dict
        A dictionary where keys are normalization method names and values are tuples
        containing the normalization function and the sample limit.
    """
    return {'EMSC': (normalize_emsc, 200),
            'SNV': (normalize_snv, 16_000),
            'Area': (normalize_area, 16_000),
            'Trapezoidal rule area': (normalize_trapz_area, 16_000),
            'Max intensity': (normalize_max, 16_000),
            'Min-max intensity': (normalize_minmax, 16_000)
            }


def peak_shapes_params() -> dict:
    """
    Returns a dictionary of peak shape functions and additional parameters.

    Returns
    -------
    dict
        A dictionary where keys are peak shape names and values are dictionaries
        containing the peak shape function and additional parameters.
    """
    return {'Gaussian': {'func': gaussian},
            'Split Gaussian': {'func': split_gaussian, 'add_params': ['dx_left']},
            'Skewed Gaussian': {'func': skewed_gaussian, 'add_params': ['gamma']},
            'Lorentzian': {'func': lorentzian},
            'Split Lorentzian': {'func': split_lorentzian, 'add_params': ['dx_left']},
            'Voigt': {'func': voigt, 'add_params': ['gamma']},
            'Split Voigt': {'func': split_voigt, 'add_params': ['dx_left', 'gamma']},
            'Skewed Voigt': {'func': skewed_voigt, 'add_params': ['gamma', 'skew']},
            'Pseudo Voigt': {'func': pseudovoigt, 'add_params': ['l_ratio']},
            'Split Pseudo Voigt': {'func': split_pseudovoigt, 'add_params': ['dx_left', 'l_ratio']},
            'Pearson4': {'func': pearson4, 'add_params': ['expon', 'skew']},
            'Split Pearson4': {'func': split_pearson4, 'add_params': ['dx_left', 'expon', 'skew']},
            'Pearson7': {'func': pearson7, 'add_params': ['expon']},
            'Split Pearson7': {'func': split_pearson7, 'add_params': ['dx_left', 'expon']}
            }


def smoothing_methods() -> dict[str, tuple[Callable, int]]:
    """
    Returns a dictionary of smoothing methods with their corresponding functions and sample limits.

    Returns
    -------
    dict
        A dictionary where keys are smoothing method names and values are tuples
        containing the smoothing function and the sample limit.
    """
    return {'MLESG': (smooth_mlesg, 256),
            'CEEMDAN': (smooth_ceemdan, 12),
            'EEMD': (smooth_eemd, 60),
            'EMD': (smooth_emd, 600),
            'Savitsky-Golay filter': (smooth_savgol, 10_000),
            'Whittaker smoother': (whittaker, 4_000),
            'Flat window': (smooth_flat, 12_000),
            'hanning': (smooth_window, 12_000),
            'hamming': (smooth_window, 12_000),
            'bartlett': (smooth_window, 12_000),
            'blackman': (smooth_window, 12_000),
            'kaiser': (smooth_window_kaiser, 12_000),
            'Median filter': (smooth_med_filt, 12_000),
            'Wiener filter': (smooth_wiener, 12_000),
            }


def get_optuna_params() -> dict:
    """
    Returns a dictionary of hyperparameters for Optuna optimization for various classifiers.

    Returns
    -------
    dict
        A dictionary where keys are classifier names and values are lists of hyperparameter names.
    """
    return {'LDA': [["solver", "shrinkage"]],
            'Logistic regression': [["penalty", "solver"], ["C", "l1_ratio"]],
            'SVC': [['C', 'tol']],
            'Decision Tree': [['criterion', 'max_depth'], ['splitter', 'min_samples_split'],
                              ['min_samples_leaf', 'max_features']],
            'Random Forest': [['criterion', 'max_depth'], ['min_samples_leaf', 'min_samples_split'],
                              ['max_samples', 'max_features']],
            'XGBoost': [['max_depth', 'gamma'], ['min_child_weight', 'reg_alpha'],
                        ['reg_lambda', 'subsample'], ['colsample_bytree', 'colsample_bylevel'],
                        ['colsample_bynode', 'max_delta_step']]
            }


def peak_shape_params_limits() -> dict:
    """
    Limits for additional parameters.
    """
    return {'gamma': (0., 100.), 'skew': (-100., 100.), 'l_ratio': (0., 1.),  'expon': (0.1, 100.),
            'beta': (0.1, 100.), 'alpha': (-1., 1.), 'q': (-0.5, 0.5)}
