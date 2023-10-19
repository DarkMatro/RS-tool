# Raman spectroscopy tool (RS-tool)

![splash](https://github.com/DarkMatro/RS-tool/assets/113565324/8e359ff6-0a3a-4628-b8ab-bd0364bc8bb2)

Implementation of a full cycle of biomedical machine learning classification problem.
The program is designed for a realtime full cycle of processing Raman spectra: preprocessing, decomposition of spectra, creation of classifier models for further work with new measured spectra.
New original ExModPoly (Extended ModPoly algorithm for baseline correction) and decomposition methods available.

# Installation
1. Download the installer in the Releases section and install.
2. Install Graphviz https://graphviz.org/download/. (only needed to display XGBoost trees).

# Program functionality
- Classifier training and measurement classification using models: 'LDA', 'Logistic regression', 'NuSVC', 'Nearest Neighbors', 'GPC', 'Decision Tree', 'Naive Bayes', 'Random Forest' , 'AdaBoost', 'MLP' (1 hidden layer neural network), XGBoost, Voting and Stacking.
- Data dimensionality reduction using PCA and PLS-DA methods.
- Obtaining Variable importance in projections (VIP) PLS-DA values.
- The ability to use the program to classify new measured spectra using trained models.
- Raman spectrum auto decomposition algorithm.
- Baseline correction with 42 methods: ExModPoly Poly, ModPoly, iModPoly, Penalized poly, LOESS, Quantile regression, Goldindec, AsLS, iAsLS, arPLS, airPLS, drPLS, iarPLS, asPLS, psaLSA, DerPSALSA, MPLS, iMor, MorMol, AMorMol , MPSpline, JBCD, Mixture Model, IRSQR, Corner-Cutting, RIA, Dietrich, Golotvin, Std Distribution, FastChrom, FABC.
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
- Averaged plot with confidence interval.
  
![image](https://github.com/DarkMatro/RS-tool/assets/113565324/8a683c0f-8293-46ea-9550-5da0ae9747da)
![image](https://github.com/DarkMatro/RS-tool/assets/113565324/c79fd0d1-052d-4590-8726-2060b387bc8b)
![image](https://github.com/DarkMatro/RS-tool/assets/113565324/78d2f2b0-c137-43f7-a0a8-f2a16bcf1ee1)
![image](https://github.com/DarkMatro/RS-tool/assets/113565324/62a4c385-430e-4b80-948d-7a5083274558)
![image](https://github.com/DarkMatro/RS-tool/assets/113565324/a28d64f1-c419-460d-a2a4-8167eb5a8be9)








