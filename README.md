# Raman spectroscopy processing tool (RS-tool)
Implementation of a full cycle of biomedical machine learning classification problem.
The program is designed for a full cycle of processing Raman spectra: preprocessing, decomposition of spectra, creation of classifier models for further work with new measured spectra.
New original ExModPoly (Extended ModPoly algorithm for baseline correction) and decomposition methods available.

# Installation
1. Download the installer in the Releases section and install.
2. Install Graphviz https://graphviz.org/download/. (only needed to display XGBoost trees).

# Program functionality
- Classifier training and measurement classification using models: 'LDA', 'QDA', 'Logistic regression', 'NuSVC', 'Nearest Neighbors', 'GPC', 'Decision Tree', 'Naive Bayes', 'Random Forest' , 'AdaBoost', 'MLP' (neural network), XGBoost.
- Data dimensionality reduction using PCA and PLS-DA methods.
- Obtaining Variable importance in projections (VIP) PLS-DA values.
- The ability to use the program to classify new measured spectra using trained models.
- Raman spectrum decomposition algorithm.
- Baseline correction with 42 methods: Poly, ModPoly, iModPoly, ExModPoly, Penalized poly, LOESS, Quantile regression, Goldindec, AsLS, iAsLS, arPLS, airPLS, drPLS, iarPLS, asPLS, psaLSA, DerPSALSA, MPLS, iMor, MorMol, AMorMol , MPSpline, JBCD, Mixture Model, IRSQR, Corner-Cutting, RIA, Dietrich, Golotvin, Std Distribution, FastChrom, FABC.
- Import files in .txt, .asc format as a two-dimensional array with spectra.
- Saving all data in a .zip project file. All information is converted into binary format and compressed in a zip archive with the maximum compression ratio.
- Interpolation to bring spectra with different wavelength ranges to the same nm range.
- Despike (removal of high-frequency random noise (electrical noise)).
- Conversion of spectra from the nm wavelength range to wavenumbers cm-1.
- Normalization of spectra using 6 methods: EMSC, SNV, Area, Trapezoidal rule area, Max intensity, Min-max intensity.
- Smoothing of spectra using 14 methods: MLESG, CEEMDAN, EEMD, EMD, Savitsky-Golay filter, Whittaker smoother, Flat window, Hanning window, Hamming window, Bartlett window, Blackman window, Kaiser window, Median filter, Wiener filter.
- Used line profiles for decomposition: Gaussian, Split Gaussian, Skewed Gaussian, Lorentzian, Split Lorentzian, Voigt, Split Voigt, Skewed Voigt, Pseudo Voigt, Split Pseudo Voigt, Pearson4, Split Pearson4, Pearson7, Split Pearson7.
- Optimization methods used for line decomposition: "Levenberg-Marquardt", "Least-Squares, 'Nelder-Mead', 'L-BFGS-B', 'Powell', 'Conjugate-Gradient', 'Cobyla', 'BFGS' , 'Truncated Newton', 'trust-region for constrained optimization', 'Sequential Linear Squares Programming'.
- Automatic construction of all necessary graphs for articles and the ability to save them in various formats.

![1](https://github.com/DarkMatro/RS-tool/assets/113565324/d449b6f3-fd60-4a62-afe1-941ec252d231)
![2](https://github.com/DarkMatro/RS-tool/assets/113565324/f72cea02-f66f-41ba-aad6-ea1889b2deda)
![3](https://github.com/DarkMatro/RS-tool/assets/113565324/52eb9c92-b2da-41cb-9c7d-acec6c8a246c)




