from modules.functions_baseline_correction import *
from modules.functions_classificators import do_lda, do_qda, do_lr, do_svc, do_nn, do_gpc, do_dt, do_nb, do_rf, do_ab, \
    do_mlp, do_pca, do_plsda, do_xgboost
from modules.functions_normalize import *
from modules.functions_peak_shapes import *
from modules.functions_smoothing import *


def default_values() -> dict[float | str]:
    """

    @return: dict[float | str]
        dictionary with default values
    """
    return {'cm_range_start': 310.0,
            'cm_range_end': 2000.0,
            'interval_start': 310.0,
            'interval_end': 2000.0,
            'trim_start_cm': 370.0,
            'trim_end_cm': 1780.0,
            'maxima_count_despike': 2,
            'laser_wl': 784.5,
            'despike_fwhm': 0,
            'neg_grad_factor_spinBox': 2,
            'normalizing_method_comboBox': 'EMSC Kohler',
            'smoothing_method_comboBox': 'MLESG',
            'window_length_spinBox': 5,
            'smooth_polyorder_spinBox': 2,
            'whittaker_lambda_spinBox': 1,
            'kaiser_beta': 14.0,
            'EMD_noise_modes': 1,
            'max_CCD_value': 65536,
            'EEMD_trials': 10,
            'sigma': 4,
            'EMSC_N_PCA': 8,
            'baseline_method_comboBox': 'ExModPoly',
            'guess_method_cb': 'Average',
            'lambda_spinBox': 100_000,
            'p_doubleSpinBox': 0.01,
            'eta': 0.5,
            'N_iterations': 1000,
            'polynome_degree': 8,
            'grad': 1e-7,
            'quantile': 0.05,
            'alpha_factor': 0.99,
            'cost_function': 'asymmetric_truncated_quadratic',
            'fraction': 0.2,
            'scale': 3,
            'peak_ratio': 0.5,
            'spline_degree': 3,
            'num_std': 3.0,
            'interp_half_window': 5,
            'sections': 32,
            'min_length': 2,
            'fill_half_window': 3,
            'opt_method_oer': 'asls',
            'fit_method': 'Levenberg-Marquardt',
            'max_dx_guess': 12.0,
            'average_function': 'Mean',
            'n_lines_method': 'Min',
            'max_noise_level': 30.,
            'dataset_type_cb': 'Deconvoluted',
            'test_data_ratio_spinBox': 25,
            'random_state_sb': 0,
            'mlp_layer_size_spinBox': 100,
            'feature_display_max_spinBox': 50,
            'l_ratio': .25,
            }


def peak_shapes_params() -> dict:
    """
    Using in self.peak_shapes_params in RS
    @return: dict
    """
    return {'Gaussian':
                {'func': gaussian},
            'Split Gaussian':
                {'func': split_gaussian, 'add_params': ['dx_left']},
            'Skewed Gaussian':
                {'func': skewed_gaussian, 'add_params': ['gamma']},
            'Lorentzian':
                {'func': lorentzian},
            'Split Lorentzian':
                {'func': split_lorentzian, 'add_params': ['dx_left']},
            'Voigt':
                {'func': voigt, 'add_params': ['gamma']},
            'Split Voigt':
                {'func': split_voigt, 'add_params': ['dx_left', 'gamma']},
            'Skewed Voigt':
                {'func': skewed_voigt, 'add_params': ['gamma', 'skew']},
            'Pseudo Voigt':
                {'func': pseudovoigt, 'add_params': ['l_ratio']},
            'Split Pseudo Voigt':
                {'func': split_pseudovoigt, 'add_params': ['dx_left', 'l_ratio']},
            'Pearson4':
                {'func': pearson4, 'add_params': ['expon', 'skew']},
            'Split Pearson4':
                {'func': split_pearson4, 'add_params': ['dx_left', 'expon', 'skew']},
            'Pearson7':
                {'func': pearson7, 'add_params': ['expon']},
            'Split Pearson7':
                {'func': split_pearson7, 'add_params': ['dx_left', 'expon']}
            # 'Moffat':
            #     {'func': moffat, 'add_params': ['beta']},
            # 'Split Moffat':
            #     {'func': split_moffat, 'add_params': ['dx_left', 'beta']}
            # 'Doniach':
            #     {'func': doniach, 'add_params': ['alpha']}
            # 'Breit-Wigner-Fano':
            #     {'func': bwf, 'add_params': ['q']}
            }


def peak_shape_params_limits() -> dict[str, tuple[float, float]]:
    """
    Using in self.peak_shape_params_limits
    @return: dict[str, tuple[float, float]]
    """
    return {'gamma': (0., 100.), 'skew': (-100., 100.), 'l_ratio': (0., 1.), 'expon': (0.1, 100.),
            'beta': (0.1, 100.), 'alpha': (-1., 1.), 'q': (-0.5, 0.5)}


def fitting_methods() -> dict[str]:
    """
    fitting_methods. see lmfit.minimizer
    @return: dict[str]
    """
    return {'Levenberg-Marquardt': 'leastsq',
            'Least-Squares, Trust Region Reflective method': 'least_squares',
            # 'Differential evolution': 'differential_evolution',
            # 'Brute force method': 'brute',
            # 'Basin-hopping': 'basinhopping',
            # 'Adaptive Memory Programming for Global Optimization': 'ampgo',
            'Nelder-Mead': 'nelder',
            'L-BFGS-B': 'lbfgsb',
            'Powell': 'powell',
            'Conjugate-Gradient': 'cg',
            # 'Newton-CG': 'newton',
            'Cobyla': 'cobyla',
            'BFGS': 'bfgs',
            'Truncated Newton': 'tnc',
            # 'Newton-CG trust-region': 'trust-ncg',
            # 'nearly exact trust-region': 'trust-exact',
            # 'Newton GLTR trust-region': 'trust-krylov',
            'trust-region for constrained optimization': 'trust-constr',
            # 'Dog-leg trust-region': 'dogleg',
            'Sequential Linear Squares Programming': 'slsqp',
            # 'Maximum likelihood via Monte-Carlo Markov Chain': 'emcee',
            # 'Simplicial Homology Global Optimization': 'shgo',
            # 'Dual Annealing optimization': 'dual_annealing'
            }


def baseline_methods() -> dict[tuple]:
    """
    keys using in adding items of baseline_correction_method_comboBox
    func is using in baseline correction
    limit is n_samples_limit in do_baseline_correction to switch btwn ThreadPoolExecutor and ProcessPoolExecutor
    @return: dict[tuple]
    """
    return {
        # Polynomial methods
        'Poly': (baseline_poly, 10_000),
        'ModPoly': (baseline_modpoly, 1200),
        'iModPoly': (baseline_imodpoly, 5000),
        'ExModPoly': (ex_mod_poly, 2000),
        'Penalized poly': (baseline_penalized_poly, 10_000),
        # 'LOESS': (baseline_loess, 100),
        'Quantile regression': (baseline_quant_reg, 740),
        'Goldindec': (baseline_goldindec, 160),
        # Whittaker-smoothing-based methods
        'AsLS': (baseline_asls, 190),
        'iAsLS': (baseline_iasls, 2700),
        'arPLS': (baseline_arpls, 175),
        'airPLS': (baseline_airpls, 100),
        'drPLS': (baseline_drpls, 270),
        'iarPLS': (baseline_iarpls, 830),
        'asPLS': (baseline_aspls, 330),
        'psaLSA': (baseline_psalsa, 3800),
        'DerPSALSA': (baseline_derpsalsa, 1540),
        # Morphological methods
        'MPLS': (baseline_mpls, 1000),
        'Morphological': (baseline_mor, 1000),
        'iMor': (baseline_imor, 175),
        'MorMol': (baseline_mormol, 790),
        'AMorMol': (baseline_amormol, 175),
        'Rolling Ball': (baseline_rolling_ball, 1000),
        'MWMV': (baseline_mwmv, 1000),
        'Top-hat': (baseline_tophat, 1000),
        'MPSpline': (baseline_mpspline, 750),
        'JBCD': (baseline_jbcd, 810),
        # Spline methods
        'Mixture Model': (baseline_mixture_model, 240),
        'IRSQR': (baseline_irsqr, 890),
        'Corner-Cutting': (baseline_corner_cutting, 4500),
        # Smoothing-based methods
        'Noise Median': (baseline_noise_median, 1230),
        'SNIP': (baseline_snip, 800),
        'SWiMA': (baseline_swima, 580),
        'IPSA': (baseline_ipsa, 740),
        'RIA': (baseline_ria, 560),
        # Baseline/Peak Classification methods
        'Dietrich': (baseline_dietrich, 740),
        'Golotvin': (baseline_golotvin, 1000),
        'Std Distribution': (baseline_std_distribution, 1000),
        'FastChrom': (baseline_fastchrom, 740),
        'FABC': (baseline_fabc, 770),
        # Optimizers
        'OER': (baseline_optimize_extended_range, 400),
        'Adaptive MinMax': (baseline_adaptive_minmax, 780),
        # Miscellaneous methods
        'BEaDS': (baseline_beads, 700)
    }


def baseline_parameter_defaults() -> dict[dict[float | int]]:
    return {'Poly': {'poly_deg': 5},
            'ModPoly': {'poly_deg': 5, 'grad': 1e-3, 'n_iter': 250},
            'iModPoly': {'poly_deg': 6, 'grad': 1e-3, 'n_iter': 250},
            'ExModPoly': {'poly_deg': 7, 'grad': 1e-7, 'n_iter': 300},
            'Penalized poly': {'poly_deg': 2, 'grad': 1e-3, 'n_iter': 250, 'alpha_factor': 0.99},
            'LOESS': {'poly_deg': 1, 'grad': 1e-3, 'n_iter': 10, 'fraction': 0.2, 'scale': 3.0},
            'Quantile regression': {'poly_deg': 2, 'grad': 1e-6, 'n_iter': 250, 'quantile': 0.01},
            'Goldindec': {'poly_deg': 2, 'grad': 1e-3, 'n_iter': 250, 'peak_ratio': 0.5, 'alpha_factor': 0.99},
            'AsLS': {'lam': 1e6, 'p': 1e-3, 'n_iter': 50},
            'iAsLS': {'lam': 1e6, 'p': 1e-3, 'n_iter': 50},
            'arPLS': {'lam': 1e5, 'p': 1e-6, 'n_iter': 50},
            'airPLS': {'lam': 1e6, 'p': 1e-6, 'n_iter': 50},
            'drPLS': {'lam': 1e5, 'p': 1e-6, 'n_iter': 50, 'eta': 0.5},
            'iarPLS': {'lam': 100, 'p': 1e-6, 'n_iter': 50},
            'asPLS': {'lam': 1e5, 'p': 1e-6, 'n_iter': 100},
            'psaLSA': {'lam': 1e5, 'p': 0.5, 'n_iter': 50},
            'DerPSALSA': {'lam': 1e3, 'p': 0.01, 'n_iter': 50},
            'MPLS': {'lam': 1e6, 'p': 0.0, 'n_iter': 50},
            'iMor': {'n_iter': 200, 'grad': 1e-3},
            'MorMol': {'n_iter': 250, 'grad': 1e-3},
            'AMorMol': {'n_iter': 200, 'grad': 1e-3},
            'MPSpline': {'lam': 1e4, 'p': 0.0, 'spl_deg': 3},
            'JBCD': {'n_iter': 20, 'grad': 1e-2},
            'Mixture Model': {'lam': 1e5, 'p': 1e-2, 'spl_deg': 3, 'n_iter': 50, 'grad': 1e-3},
            'IRSQR': {'lam': 100, 'quantile': 0.01, 'spl_deg': 3, 'n_iter': 100},
            'Corner-Cutting': {'n_iter': 100},
            'RIA': {'grad': 1e-6},
            'Dietrich': {'num_std': 3.0, 'poly_deg': 5, 'n_iter': 50, 'grad': 1e-3, 'interp_half_window': 5,
                         'min_length': 2},
            'Golotvin': {'num_std': 2.0, 'interp_half_window': 5, 'min_length': 2, 'sections': 32},
            'Std Distribution': {'num_std': 1.1, 'interp_half_window': 5, 'fill_half_window': 3},
            'FastChrom': {'interp_half_window': 5, 'min_length': 2, 'n_iter': 100},
            'FABC': {'lam': 1e6, 'num_std': 3.0, 'min_length': 2}
            }


def optimize_extended_range_methods() -> list[str]:
    return ['Poly', 'ModPoly', 'iModPoly', 'ExModPoly', 'Penalized_poly', 'Quant_Reg', 'Goldindec', 'AsLS', 'iAsLS',
            'airPLS', 'arPLS', 'drPLS', 'iarPLS', 'asPLS', 'psaLSA', 'DerPSALSA', 'MPLS', 'MPSpline', 'Mixture_model',
            'IRSQR', 'Dietrich', 'Golotvin', 'Std_distribution', 'FastChrom', 'CWT_BR', 'FABC']


def smoothing_methods() -> dict[tuple]:
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


def normalize_methods() -> dict[str, tuple]:
    """
    keys using in adding items of normalizing_method_comboBox
    func is using in normalize procedure
    limit is n_samples_limit to switch btwn ThreadPoolExecutor and ProcessPoolExecutor
    @return: dict[tuple]
    """
    return {'EMSC': (normalize_emsc, 200),
            'SNV': (normalize_snv, 16_000),
            # 'SNV+': (normalize_snv_plus, 16_000),
            'Area': (normalize_area, 16_000),
            # 'Area+': (normalize_area_plus, 16_000),
            'Trapezoidal rule area': (normalize_trapz_area, 16_000),
            'Max intensity': (normalize_max, 16_000),
            'Min-max intensity': (normalize_minmax, 16_000)
            }


def classificator_funcs() -> dict[str, callable]:
    return {'LDA': do_lda, 'QDA': do_qda, 'Logistic regression': do_lr, 'NuSVC': do_svc, 'Nearest Neighbors': do_nn,
            'GPC': do_gpc, 'Decision Tree': do_dt, 'Naive Bayes': do_nb, 'Random Forest': do_rf, 'AdaBoost': do_ab,
            'MLP': do_mlp, 'XGBoost': do_xgboost, 'PCA': do_pca, 'PLS-DA': do_plsda}


def program_version() -> str:
    return 'ver. 1.0.04 '
