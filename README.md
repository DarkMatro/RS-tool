# Raman spectroscopy processing tool (RS-tool)
Программа предназначена для обработки Рамановских спектров, декомпозиции спектров, создание моделей-классификаторов для дальнейшей работы с новоми измеренными спектрами.

# Установка
1. Скачать установщик в разделе Releases, установить.
2. Установить Graphviz https://graphviz.org/download/. (нужно только для отображения деревьев XGBoost).

# Функционал программы
- Импорт файлов в формате .txt, .asc в виде двумерного массива со спектрами.
- Импорт файлов в формате .txt, .asc в виде двумерного массива со спектрами.
- Сохранение всех данных в файле проекта .zip. Вся информация переводится в бинарный формат и сжимается в zip архиве с максимальной степенью сжатия.
- Интерполяция для приведения спектров с различными диапазонами длин волн к одному диапазону нм.
- Despike (удаление высокочастотных случайных шумов (наводок с электросети)).
- Конвертация спектров из диапазона длин волн нм в волновые числа см-1 .
- Нормализация спектров 6 методами: EMSC, SNV, Area, Trapezoidal rule area, Max intensity, Min-max intensity.
- Сглаживание спектров 14 методами: MLESG, CEEMDAN, EEMD, EMD, Savitsky-Golay filter, Whittaker smoother, Flat window, Hanning window, Hamming window, Bartlett window, Blackman window, Kaiser window, Median filter, Wiener filter.
- Коррекция базовой линии 42 методами: Poly, ModPoly, iModPoly, ExModPoly, Penalized poly, LOESS, Quantile regression, Goldindec, AsLS, iAsLS, arPLS, airPLS, drPLS, iarPLS, asPLS, psaLSA, DerPSALSA, MPLS, iMor, MorMol, AMorMol, MPSpline, JBCD, Mixture Model, IRSQR, Corner-Cutting, RIA, Dietrich, Golotvin, Std Distribution, FastChrom, FABC.
- Алгоритм декомпозиции спектра КР.
- Используемые профили линий для декомпозиции: Gaussian, Split Gaussian, Skewed Gaussian, Lorentzian, Split Lorentzian, Voigt, Split Voigt, Skewed Voigt, Pseudo Voigt, Split Pseudo Voigt, Pearson4, Split Pearson4, Pearson7, Split Pearson7.
- Используемые при декомпозиции линий методы оптимизации: "Levenberg-Marquardt",  "Least-Squares, 'Nelder-Mead', 'L-BFGS-B', 'Powell', 'Conjugate-Gradient', 'Cobyla', 'BFGS', 'Truncated Newton', 'trust-region for constrained optimization', 'Sequential Linear Squares Programming'.
- Обучение классификаторов и классификация измерений с помощью моделей: 'LDA', 'QDA', 'Logistic regression', 'NuSVC', 'Nearest Neighbors', 'GPC', 'Decision Tree', 'Naive Bayes', 'Random Forest', 'AdaBoost', 'MLP' (нейросеть), XGBoost.
- Снижение размерности данных методами PCA и PLS-DA.
- Получения значений Variable importance in projections (VIP) PLS-DA.
- Возможность использования программы для классификации по обученным моделям новых измеренных спектров.
- Автоматическое построение всех необходимых графиков для статей и возможность сохранения их в различных форматах.

![1](https://github.com/DarkMatro/RS-tool/assets/113565324/d449b6f3-fd60-4a62-afe1-941ec252d231)
![2](https://github.com/DarkMatro/RS-tool/assets/113565324/f72cea02-f66f-41ba-aad6-ea1889b2deda)
![3](https://github.com/DarkMatro/RS-tool/assets/113565324/52eb9c92-b2da-41cb-9c7d-acec6c8a246c)




