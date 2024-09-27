# Raman spectroscopy tool (RS-tool)

![splash](https://github.com/user-attachments/assets/e42bdd44-82d6-4129-93fd-d134153d8566)

Implementation of a full cycle of biomedical machine learning classification problem.
The program is designed for a realtime full cycle of processing Raman spectra: preprocessing, decomposition of spectra, creation of classifier models for further work with new measured spectra.
New original ExModPoly (Extended ModPoly algorithm for baseline correction) and decomposition methods available.

# Installation
1. Download the installer in the Releases section and install.
2. Install Graphviz https://graphviz.org/download/. (only needed to display XGBoost trees).

# Program functionality
- Import of files in .txt, .asc format as a two-dimensional array with spectra.
- Saving all data in a .zip project file. All information is converted to binary format and compressed in a zip archive with the maximum compression ratio.
- Interpolation to bring spectra with different wavelength ranges to one nm range.
- Despike (removal of high-frequency random noise (electrical interference)).
- Conversion of spectra from the nm wavelength range to wavenumbers cm-1.
- Normalization of spectra by 6 methods: EMSC, SNV, Area, Trapezoidal rule area, Max intensity, Min-max intensity.
- Smoothing of spectra by 14 methods: MLESG, CEEMDAN, EEMD, EMD, Savitsky-Golay filter, Whittaker smoother, Flat window, Hanning window, Hamming window, Bartlett window, Blackman window, Kaiser window, Median filter, Wiener filter.
- Baseline correction using 42 methods:: Poly, ModPoly, iModPoly, ExModPoly, Penalized poly, LOESS, Quantile regression, Goldindec, AsLS, iAsLS, arPLS, airPLS, drPLS, iarPLS, asPLS, psaLSA, DerPSALSA, MPLS, iMor, MorMol, AMorMol, MPSpline, JBCD, Mixture Model, IRSQR, Corner-Cutting, RIA, Dietrich, Golotvin, Std Distribution, FastChrom, FABC.
- Raman spectrum decomposition algorithm.
- Line profiles used for decomposition: Gaussian, Split Gaussian, Skewed Gaussian, Lorentzian, Split Lorentzian, Voigt, Split Voigt, Skewed Voigt, Pseudo Voigt, Split Pseudo Voigt, Pearson4, Split Pearson4, Pearson7, Split Pearson7.
- Optimization methods used in line decomposition: "Levenberg-Marquardt",  "Least-Squares, 'Nelder-Mead', 'L-BFGS-B', 'Powell', 'Conjugate-Gradient', 'Cobyla', 'BFGS', 'Truncated Newton', 'trust-region for constrained optimization', 'Sequential Linear Squares Programming', 'Basin-hopping', 'Adaptive Memory Programming for Global Optimization', 'Dual Annealing optimization', 'Simplicial Homology Global Optimization'
- Training classifiers and classifying dimensions using models: 'LDA',  'Logistic regression', 'SVC', 'Decision Tree', 'Random Forest',  XGBoost.
- Reducing the dimensionality of data using the method PCA.
- Possibility of using the program for classification of new measured spectra according to trained models.
- Automatic generation of all necessary graphs for articles and the ability to save them in various formats.
- Intuitive graphical interface.

# Citation
Decomposition Method for Raman Spectra of Dentine. DOI: 10.18287/JBPE24.10.030303. https://jbpe.ssau.ru/index.php/JBPE/article/view/9074

![image](https://github.com/user-attachments/assets/f988dd8c-1b69-41e4-b73b-ee8ea9cdb260)
![image](https://github.com/user-attachments/assets/f43bdd02-21f9-483f-8d9a-55b4fe0d5098)
![image](https://github.com/user-attachments/assets/dd512e53-4859-4f99-8128-3295dd7ebf3e)
![image](https://github.com/user-attachments/assets/f41f8a1b-8eff-47f2-87f9-20b658c3eb04)
![image](https://github.com/user-attachments/assets/919b86eb-d011-4b50-881d-e9bdc80030fe)









