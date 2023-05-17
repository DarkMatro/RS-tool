# Raman spectroscopy processing tool (RS-tool)
Программа предназначена для обработки Рамановских спектров, разделения спектров на элементарные линии, создание моделей-классификаторов.

# Установка
1. Скачать установщик в разделе Releases, установить.
2. Установить Graphviz https://graphviz.org/download/.

# Функционал программы
- Импорт файлов в формате .txt, .asc в виде двумерного массива со спектрами.
- Сохранение всех данных в файле проекта .zip. Вся информация переводится в бинарный формат и сжимается в zip архиве с максимальной степенью сжатия.
- Интерполяция для приведения спектров с различными диапазонами длин волн к одному диапазону нм.
- Despike (удаление случайных шумов).
- Конвертация спектров из диапазона длин волн нм в волновые числа см-1 .
- Нормализация спектров 6 методами: EMSC, SNV, Area, Trapezoidal rule area, Max intensity, Min-max intensity.
- Сглаживание спектров 14 методами: MLESG, CEEMDAN, EEMD, EMD, Savitsky-Golay filter, Whittaker smoother, Flat window, Hanning window, Hamming window, Bartlett window, Blackman window, Kaiser window, Median filter, Wiener filter.
Коррекция базовой линии 42 методами: Poly, ModPoly, iModPoly, iModPoly+, Penalized poly, LOESS, Quantile regression, Goldindec, AsLS, iAsLS, arPLS, airPLS, drPLS, iarPLS, asPLS, psaLSA, DerPSALSA, MPLS, iMor, MorMol, AMorMol, MPSpline, JBCD, Mixture Model, IRSQR, Corner-Cutting, RIA, Dietrich, Golotvin, Std Distribution, FastChrom, FABC.
- Разделение спектра на сумму элементарных линий. 
- Используемые линии: Gaussian, Split Gaussian, Skewed Gaussian, Lorentzian, Split Lorentzian, Voigt, Split Voigt, Skewed Voigt, Pseudo Voigt, Split Pseudo Voigt, Pearson4, Split Pearson4, Pearson7, Split Pearson7.
- Используемые при разделении линий методы оптимизации: "Levenberg-Marquardt",  "Least-Squares, Trust Region Reflective method", 'Differential evolution', 'Basin-hopping', 'Adaptive Memory Programming for Global Optimization', 'Nelder-Mead', 'L-BFGS-B', 'Powell', 'Conjugate-Gradient', 'BFGS', 'Truncated Newton', 'trust-region for constrained optimization', 'Sequential Linear Squares Programming', 'Maximum likelihood via Monte-Carlo Markov Chain', 'Dual Annealing optimization'.
- Обучение классификаторов и классификация измерений с помощью моделей: 'LDA', 'QDA', 'Logistic regression', 'NuSVC', 'Nearest Neighbors', 'GPC', 'Decision Tree', 'Naive Bayes', 'Random Forest', 'AdaBoost', 'MLP' (нейросеть), 'XGBoost'.
- Снижение размерности данных методами PCA и PLS-DA.
- Получения значений Variable importance in projections (VIP) PLS-DA.
- Возможность использования программы для классификации по обученным моделям новых измерений спектров.
- Автоматическое построение всех необходимых графиков для статей и возможность сохранения их в различных форматах.

![drex_1_4__vkladka_preprocessing_custom](https://github.com/DarkMatro/RS-tool/assets/113565324/2606f84c-ff8c-472e-a838-20f10704c6b6)
![drex_1_5__vkladka_fitting_custom](https://github.com/DarkMatro/RS-tool/assets/113565324/50092ec4-1f8a-43a4-9dfe-9193d94aefeb)
![drex_1_7__vkladka_stat_analysis_custom_3](https://github.com/DarkMatro/RS-tool/assets/113565324/b79b219d-c83c-4bd2-8d43-a05b4c119b4a)
