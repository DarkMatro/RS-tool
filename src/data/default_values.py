from typing import Callable

from src.stages.fitting.functions.peak_shapes import gaussian, split_gaussian, skewed_gaussian, \
    lorentzian, split_lorentzian, \
    voigt, split_voigt, skewed_voigt, pseudovoigt, split_pseudovoigt, pearson4, split_pearson4, \
    pearson7, \
    split_pearson7
from src.stages.ml.functions.hyperopt import objective_lda
from src.stages.preprocessing.functions.baseline_correction import baseline_correct, ex_mod_poly, \
    baseline_asls, baseline_arpls, \
    baseline_airpls, baseline_drpls
from src.stages.preprocessing.functions.normalization.normalization import normalize_emsc, \
    normalize_snv, normalize_area, normalize_trapz_area, \
    normalize_max, normalize_minmax
from src.stages.preprocessing.functions.smoothing.functions_smoothing import smooth_mlesg, \
    smooth_ceemdan, smooth_eemd, smooth_emd, smooth_savgol, whittaker, \
    smooth_flat, smooth_window, smooth_window_kaiser, smooth_med_filt, smooth_wiener
from src.stages.ml.functions.fit_classificators import (fit_lda_clf, fit_lr_clf, \
                                                        fit_svc_clf, fit_nn_clf, fit_nb_clf,
                                                        fit_dt_clf, fit_rf_clf, fit_xgboost_clf,
                                                        fit_pca, fit_lda)

def baseline_parameter_defaults() -> dict[str, dict[str, int | float] ]:
    return {'Poly': {'polynome_degree': 5},
            'ModPoly': {'polynome_degree': 5, 'tol': 1e-3, 'max_iter': 250},
            'iModPoly': {'polynome_degree': 6, 'tol': 1e-3, 'max_iter': 250, 'num_std': 0.},
            'ExModPoly': {'polynome_degree': 7, 'tol': 1e-6, 'max_iter': 100, 'quantile': 1e-5, 'scale': .5, 'num_std': 3.,
                          'half_window': 2},
            'Penalized poly': {'polynome_degree': 2, 'tol': 1e-3, 'max_iter': 250, 'alpha_factor': 0.99999},
            'LOESS': {'polynome_degree': 1, 'tol': 1e-3, 'max_iter': 10, 'fraction': 0.2, 'scale': 3.0},
            'Quantile regression': {'polynome_degree': 2, 'tol': 1e-6, 'max_iter': 250, 'quantile': 0.01},
            'Goldindec': {'polynome_degree': 2, 'tol': 1e-3, 'max_iter': 250, 'peak_ratio': 0.5, 'alpha_factor': 0.99},
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
            'Mixture Model': {'lambda': 100000, 'p': 1e-2, 'spline_degree': 3, 'max_iter': 50, 'tol': 1e-3},
            'IRSQR': {'lambda': 100, 'quantile': 0.01, 'spline_degree': 3, 'max_iter': 100, 'tol': 1e-6},
            'Corner-Cutting': {'max_iter': 100},
            'Noise Median': {'half_window': 5},
            'IPSA': {'max_iter': 500},
            'RIA': {'tol': 1e-2, 'max_iter': 500},
            'Dietrich': {'num_std': 3.0, 'polynome_degree': 5, 'max_iter': 50, 'tol': 1e-3, 'half_window': 5,
                         'min_length': 2},
            'Golotvin': {'num_std': 2.0, 'half_window': 5, 'min_length': 2, 'sections': 32},
            'Std Distribution': {'num_std': 1.1, 'half_window': 5},
            'FastChrom': {'half_window': 5, 'min_length': 2, 'max_iter': 100},
            'FABC': {'lambda': 1000000, 'num_std': 3.0, 'min_length': 2}
            }


def scoring_metrics() -> list[str]:
    return ['precision_macro', 'precision_micro', 'precision_samples', 'precision', 'average_precision',
            'precision_weighted',
            'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted',
            'accuracy', 'top_k_accuracy', 'balanced_accuracy',
            'f1', 'f1_macro', 'f1_micro', 'f1_weighted', 'f1_samples',
            'roc_auc_ovo_weighted', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc', 'roc_auc_ovr',
            'matthews_corrcoef', 'explained_variance',
            'v_measure_score', 'max_error'
                               'neg_log_loss', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error',
            'neg_brier_score',
            'neg_mean_poisson_deviance', 'neg_mean_absolute_error', 'neg_mean_squared_log_error',
            'neg_root_mean_squared_error', 'neg_negative_likehood_ratio' 'neg_median_absolute_error',
            'neg_mean_gamma_deviance',
            'jaccard_macro', 'jaccard_micro', 'jaccard', 'jaccard_samples', 'jaccard_weighted',
            'adjusted_mutual_info_score', 'normalized_mutual_info_score', 'mutual_info_score',
            'completeness_score', 'fowlkes_mallows_score', 'homogeneity_score',
            'rand_score', 'r2', 'adjusted_rand_score', 'positive_likehood_ratio']


def baseline_methods() -> dict[str, tuple[Callable, int]]:
    """
    keys using in adding items of baseline_correction_method_comboBox
    func is using in baseline correction
    limit is n_samples_limit in do_baseline_correction to switch btwn ThreadPoolExecutor and ProcessPoolExecutor
    @return: dict[tuple]
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
    return {'LDA': fit_lda, 'Logistic regression': fit_lr_clf, 'NuSVC': fit_svc_clf,
            'Nearest Neighbors': fit_nn_clf, 'Naive Bayes': fit_nb_clf, 'Decision Tree': fit_dt_clf,
            'Random Forest': fit_rf_clf, 'XGBoost': fit_xgboost_clf, 'PCA': fit_pca}

def objectives() -> dict[str, callable]:
    return {'LDA': objective_lda}


def normalize_methods() -> dict[str, tuple]:
    """
    keys using in adding items of normalizing_method_comboBox
    func is using in normalize procedure
    limit is n_samples_limit to switch btwn ThreadPoolExecutor and ProcessPoolExecutor
    @return: dict[tuple]
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
    Using in self.peak_shapes_params in RS
    keys of dict MUST BE EQUAL TO peak_shape_names() in default_values.py

    @return: dict
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
    keys using in adding items of smoothing_method_comboBox
    func is using in smoothing procedure
    limit is n_samples_limit to switch btwn ThreadPoolExecutor and ProcessPoolExecutor
    @return: dict[tuple]
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
