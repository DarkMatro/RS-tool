"""
   # h = x_axis_new[2] - x_axis_new[1]
   # h_old = x_axis_old[2] - x_axis_old[1]
   # first_x_new = x_axis_new[0]
   # first_x_old = x_axis_old[0]
   # value_to_insert = x_axis_new[-1]
   # # for i in range(4):  # добавим в конец 3 значения для расчета для последних точек, потом убираются
   # #     x_axis_new = np.insert(x_axis_new, x_axis_new.shape, value_to_insert, axis=0)
   # y_axis_new = np.zeros(x_axis_new.shape)
   # inaccuracy = abs(h - h_old) * x_axis_new.shape[0]
   # m = inaccuracy / h
   # lim = x_axis_new.shape[0] / (2 * m)
   # lim = roundup(lim)
   # first_zero_idx = 0
   # y_last_old = y_axis_old[-1]
   # b = 0
   # for i in range(y_axis_new.size):
   #     index_old = find_nearest_idx(x_axis_old, x_axis_new[i])
   #     y_axis_new[i] = y_axis_old[index_old]
   # if lim == 0:
   #
   #         на спектр размером 1911 уходит ~22 мс, на все остальное тут уходит + 5 мс.
   #         этот цикл нужен чтобы правильно заполнить Y по соответствующему X
   #         И нужен из предположения что шаг в исходном и новом диапазоне нм отличается
   #         если просто определить первый совпадающий элемент и дальше по порядку накидывать в новый массив
   #         без проверки что x_old < x_new, ошибка может накопиться и порядок будет не тот
   #         например шаг в новом диапазоне 0.12585, в старом 0.12546, разница 0.00039.
   #         0.00039*1911 = 0.74529 (inaccuracy), накопившаяся ошибка больше чем в 5.96 (m) раз больше шага.
   #         шаг собъется уже на 320 элементе, а к концу сдвиг будет на 6 элементов
   #         Поэтому берем число 320, делим на 2 и округляем до сотен, получаем 100
   #         Каждые 100 элементов уточняемся с помощью find_nearest_idx,
   #         и остальные 99 элементов докидываем по порядку для оптимизации.
   #         C 22 мс так получается сократить вычисления до 1 мс
   #
   #     for i in range(y_axis_new.size):
   #         index_old = find_nearest_idx(x_axis_old, x_axis_new[i])
   #         y_axis_new[i] = y_axis_old[index_old]
   # else:
   #     lim = 100
   #     a = (y_axis_new.size // lim)
   #     for i in range(a + 1):
   #         b = i * lim
   #         index_old = find_nearest_idx(x_axis_old, x_axis_new[b])
   #         if first_x_new < first_x_old and index_old == 0:
   #             b = find_nearest_idx(x_axis_new, first_x_old) + 1
   #         elements_left = lim
   #         if i != a and index_old + lim > y_axis_old.size:
   #             first_zero_idx = b
   #             break
   #         elif i == a:
   #             elements_left = y_axis_new.size - b
   #         y_axis_new[b:b + elements_left] = y_axis_old[index_old:index_old + elements_left]
   # #  обработка нулей в начале
   # if first_x_new < first_x_old:
   #     idx = find_nearest_idx(x_axis_new, first_x_old) + 1
   #     for i in range(idx):
   #         if y_axis_new[i] == 0:
   #             y_axis_new[i] = y_axis_old[0]
   # #  обработка нулей с конца
   # if first_zero_idx > 0:
   #     for i in range(b, y_axis_new.size, 1):
   #         if y_axis_new[i] == 0 and y_axis_new[i - 1] != y_last_old:
   #             index_old = find_nearest_idx(x_axis_old, x_axis_new[i])
   #             y_axis_new[i] = y_axis_old[index_old]
   #         elif y_axis_new[i - 1] == y_last_old:
   #             y_axis_new[i] = y_last_old

   # if first_x_new < first_x_old:
   #     x_old = first_x_old
   #     x_new = find_nearest(x_axis_new, first_x_old, take_left_value=False)
   #     t = (x_new - x_old) / h
   # else:
   #     x_new = first_x_new
   #     x_old = find_nearest(x_axis_old, x_new, take_left_value=True)
   #     t = (x_new - x_old) / h
   # for i in range(y_axis_new.size - 4):
   #     x_new = x_axis_new[i]
   #     x_old = find_nearest(x_axis_old, x_new, take_left_value=True)
   #     t = (x_new - x_old) / h_old
   #     y0 = y_axis_new[i]
   #     y01 = y_axis_new[i + 1]
   #     y02 = y_axis_new[i + 2]
   #     y03 = y_axis_new[i + 3]
   #     y04 = y_axis_new[i + 4]
   #     dy0 = y01 - y0
   #     dy1 = y02 - y01
   #     dy2 = y03 - y02
   #     d2y0 = dy1 - dy0
   #     d2y1 = dy2 - dy1
   #     d3y0 = d2y1 - d2y0
   #     dy3 = y04 - y03
   #     d2y2 = dy3 - dy2
   #     d3y1 = d2y2 - d2y1
   #     d4y0 = d3y1 - d3y0
   #     new_y_value = y0 + dy0 * t + (d2y0 / 2) * t * (t - 1) + (d3y0 / 6) * t * (t - 1) * (t - 2) + (d4y0 / 24) * t * (t - 1) * (t - 2) * (t - 3)
   #     y_axis_new[i] = new_y_value
   # return_array = np.vstack((x_axis_new[:-4], y_axis_new[:-4])).T
   """

# if os.system("cl.exe"):
#     cl_path = find_cl_exe_path()
#     if cl_path != '':
#         i = 0
#         while os.system("cl.exe"):
#             os.environ[
#                 'PATH'] += ';' + str(cl_path[i])
#             i += 1
# if os.system("cl.exe"):
#     os.environ['PATH'] += ';' + \
#                           r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29." \
#                           r"30133\bin\Hostx64\x64"
# if os.system("cl.exe"):
#     raise RuntimeError("cl.exe still not found, path probably incorrect")

# CUDA_ON = cuda.is_available()

# ----------------------------------------------------------------------------------------------------------------------


# def create_fit_function(line_types: list[tuple]):
#     """
#     Create fit function for scipy curve_fit by code because it can be very various lines amount and types
#     def in def is necessary because there is no *args and **kwargs to send line_types parameter in curve_fit func
#
#     Parameters
#     ---------
#     line_types : list[tuple[func, int]]
#         func is function to calculate line (see self.deconv_line_params), int is a number of line parameters
#
#     Returns
#     -------
#     out : function
#         Fit function for scipy curve_fit
#
#     Examples
#     --------
#     >>> create_fit_function(line_types)
#     <function create_fit_function.<locals>.f at 0x000001D95C4D4EE0>
#     """
#     # first 3 parameters are always: a, x0, dx.
#     def f(x, *args):
#         y = np.zeros_like(x)
#         i = 0
#         for func, params_num in line_types:
#             if params_num == 3:
#                 y += func(x, args[i], args[i + 1], args[i + 2])
#             if params_num == 4:
#                 y += func(x, args[i], args[i + 1], args[i + 2], args[i + 3])
#             if params_num == 5:
#                 y += func(x, args[i], args[i + 1], args[i + 2], args[i + 3], args[i + 4])
#             if params_num == 6:
#                 y += func(x, args[i], args[i + 1], args[i + 2], args[i + 3], args[i + 4], args[i + 5])
#             i += params_num
#         return y
#     return f

# ------------------------------------------------------------------------------------------------------------------

# def find_cl_exe_path() -> list:
#     result = []
#     mvs_path = Path('C:\Program Files (x86)\Microsoft Visual Studio')
#     if Path('C:\Program Files\Microsoft Visual Studio').exists():
#         mvs_path = Path('C:\Program Files\Microsoft Visual Studio')
#     elif not Path('C:\Program Files (x86)\Microsoft Visual Studio').exists():
#         return ''
#     for g in mvs_path.glob('**/*cl.exe'):
#         if g.name == 'cl.exe':
#             result.append(g.parent)
#     return result

# ------------------------------------------------------------------------------------------------------------------

# def get_deconv_curve_for_plot(full_array_y: np.ndarray, x_axis: np.ndarray) -> np.ndarray:
#     """
#     Leave only y_data where amplitude > y_max * 1e-
#     @param full_array_y: y data
#     @param x_axis: x data
#     @return: 2d array with x|y data
#     """
#     y_max = np.max(full_array_y)
#     limit = y_max * 1e-4
#     arg_where = np.argwhere(full_array_y > limit).T
#     x = x_axis
#     y = full_array_y
#     if arg_where.size != 0:
#         y = full_array_y[arg_where]
#         x = x_axis[arg_where]
#     return np.vstack((x, y)).T


# @memory_profiler.profile()
# yappi.set_clock_type("cpu")
# yappi.start()
# yappi.get_func_stats().print_all()
# yappi.get_thread_stats().print_all()

# before = datetime.now()
# print(datetime.now() - before)