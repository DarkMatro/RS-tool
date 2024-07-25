from logging import error
from os import getpid

import numpy as np
from psutil import Process
from pyperclip import copy as pyperclip_copy
from qtpy.QtGui import QColor

from sklearn.metrics import mean_squared_log_error

from qfluentwidgets import MessageBox


# region RS


def invert_color(color: str) -> str:
    rgb = QColor(color).getRgb()
    new_r = 255 - rgb[0]
    new_g = 255 - rgb[1]
    new_b = 255 - rgb[2]
    return QColor(new_r, new_g, new_b, rgb[3]).name()


def get_memory_used() -> float:
    return Process(getpid()).memory_info().rss / 1024 ** 2


# endregion


def calculate_vips(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T, t), q.T), q)).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (np.matmul(s.T, weight)) / total_s)
    return vips


def show_error_msg(exc_type, exc_value, exc_tb, parent=None):
    msg = MessageBox(str(exc_type), str(exc_value), parent, {'Ok'})
    msg.setInformativeText('For full text of error go to %appdata%/RS-Tool/log.log' + '\n' + exc_tb)
    print(exc_tb)
    error(exc_tb)
    pyperclip_copy(exc_tb)
    msg.exec()


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    The Root Mean Squared Log Error (RMSLE) metric
    Логаритмическая ошибка средней квадратичной ошибки
    """
    try:
        return np.sqrt(mean_squared_log_error(y_true, y_pred))
    except:
        return None
