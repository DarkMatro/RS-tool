# pylint: disable=no-name-in-module, too-many-lines, invalid-name, import-error
"""
This module provides utility functions for plotting and color generation in a PyQt application.

Functions:
    random_rgb
        Generate a random RGB color code.

    initial_stat_plot
        Clears the current axis of the given plot widget's canvas.

    initial_shap_plot
        Clears the current figure and axis of the given plot widget's canvas.

    get_curve_plot_data_item
        Creates and returns a `PlotDataItem` with the specified data, color, style, and width.
"""

import numpy as np
from numpy.random import default_rng
from pyqtgraph import PlotDataItem
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor


def random_rgb() -> int:
    """
    Generate a random RGB color code.

    Returns
    -------
    rgb : int
        Random 9-digit RGB code. Example: 471369080
    """
    rnd_gen = default_rng()
    rnd_rgb = rnd_gen.random() * 1e9
    rgb = int(rnd_rgb)
    return rgb


def initial_stat_plot(plot_widget) -> None:
    """
    Clears the current axis of the given plot widget's canvas.

    Parameters
    ----------
    plot_widget : QWidget
        The plot widget whose canvas is to be cleared.
    """
    plot_widget.canvas.gca().cla()
    plot_widget.canvas.draw()


def initial_shap_plot(plot_widget) -> None:
    """
    Clears the current figure and axis of the given plot widget's canvas.

    Parameters
    ----------
    plot_widget : QWidget
        The plot widget whose canvas is to be cleared.
    """
    plot_widget.canvas.gca().cla()
    plot_widget.canvas.figure.clf()
    plot_widget.canvas.draw()


def get_curve_plot_data_item(n_array: np.ndarray,
                             color: QColor | str, name: str = "",
                             style: Qt.PenStyle = Qt.PenStyle.SolidLine, width: int = 2,
                             ) -> PlotDataItem:
    """
    Creates and returns a `PlotDataItem` with the specified data, color, style, and width.

    Parameters
    ----------
    n_array : np.ndarray
        Array of shape (n, 2) where n is the number of data points. The first column is x-values and
        the second column is y-values.
    color : QColor or str
        Color of the plot line.
    name : str, optional
        Name of the curve. Default is an empty string.
    style : Qt.PenStyle, optional
        Style of the plot line. Default is `Qt.PenStyle.SolidLine`.
    width : int, optional
        Width of the plot line. Default is 2.

    Returns
    -------
    PlotDataItem
        A `PlotDataItem` instance with the specified properties.
    """
    curve = PlotDataItem(skipFiniteCheck=True, name=name)
    curve.setData(x=n_array[:, 0], y=n_array[:, 1], skipFiniteCheck=True)
    curve.setPen(color, width=width, style=style)
    return curve
