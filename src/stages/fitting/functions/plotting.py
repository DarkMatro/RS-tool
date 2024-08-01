# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides various utility functions for plotting and managing plot items
such as ROI (Region of Interest) and curve styles. It includes functions for setting
ROI size and position, retrieving data items by index, updating curve styles, obtaining
line parameters, creating deconvolution curves, and generating random line styles.

Functions
---------
set_roi_size_pos
    Set the size and position of the ROI based on provided parameters.
set_roi_size
    Set the size of the ROI.
set_roi_pos
    Set the position of the ROI.
deconvolution_data_items_by_idx
    Retrieve curve and ROI data items by index.
update_curve_style
    Update the style of a curve based on the provided style dictionary.
all_lines_parameters
    Get all line parameters from the given multi-index dataframe.
get_curve_for_deconvolution
    Create a PlotCurveItem for deconvolution based on provided data and style.
random_line_style
    Generate a random line style dictionary.
curve_pen_brush_by_style
    Create pen and brush objects based on the provided style dictionary.
"""

import numpy as np
from pyqtgraph import ROI, PlotCurveItem, mkPen, mkBrush
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QColor

from src.data.default_values import peak_shapes_params
from src.data.plotting import random_rgb


def set_roi_size_pos(params: tuple[float, float, float], roi: ROI, update: bool = True) -> None:
    """
    Set the size and position of the ROI based on provided parameters.

    Parameters
    ----------
    params : tuple[float, float, float]
        Parameters containing amplitude, position, and width.
    roi : ROI
        Region of Interest object to be modified.
    update : bool, optional
        Whether to update the ROI immediately, by default True.
    """
    a, x0, dx = params
    set_roi_size(dx, a, roi, update=update)
    set_roi_pos(x0, roi)


def set_roi_size(x: float, y: float, roi: ROI, center=None, finish: bool = False,
                 update: bool = True) -> None:
    """
    Set the size of the ROI.

    Parameters
    ----------
    x : float
        Width of the ROI.
    y : float
        Height of the ROI.
    roi : ROI
        Region of Interest object to be modified.
    center : optional
        Center position for resizing, by default None.
    finish : bool, optional
        Whether to finish the resize operation, by default False.
    update : bool, optional
        Whether to update the ROI immediately, by default True.
    """
    if center is None:
        center = [0, 1]
    new_size = QPointF()
    new_size.setX(x)
    new_size.setY(y)
    roi.setSize(new_size, center, finish=finish, update=update)


def set_roi_pos(x0: float, roi: ROI) -> None:
    """
    Set the position of the ROI.

    Parameters
    ----------
    x0 : float
        X-coordinate of the ROI position.
    roi : ROI
        Region of Interest object to be modified.
    """
    new_pos = QPointF()
    new_pos.setX(x0)
    new_pos.setY(0.0)
    roi.setPos(new_pos)


def deconvolution_data_items_by_idx(idx: int, data_items: list) -> tuple[PlotCurveItem, ROI] | None:
    """
    Retrieve curve and ROI data items by index.

    Parameters
    ----------
    idx : int
        Index of the data item to retrieve.
    data_items : list
        of data items to search through.

    Returns
    -------
    tuple[PlotCurveItem, ROI] | None
        The curve and ROI data items if found, otherwise None.
    """
    if len(data_items) == 0:
        return None
    curve = None
    roi = None
    for i in data_items:
        if i.name() == idx:
            curve = i
            roi = i.parentItem()
            break
    return curve, roi


def update_curve_style(idx: int, style: dict, data_items: list) -> None:
    """
    Update the style of a curve based on the provided style dictionary.

    Parameters
    ----------
    idx : int
        Index of the curve to be updated.
    style : dict
        Dictionary containing the style parameters.
    data_items : list
        of data items containing the curves.
    """
    pen, brush = curve_pen_brush_by_style(style)
    items_matches = deconvolution_data_items_by_idx(idx, data_items)
    if items_matches is None:
        return
    curve, _ = items_matches
    if curve is None:
        return
    curve.setPen(pen)
    curve.setBrush(brush)


def all_lines_parameters(line_indexes: list[int], mw, filename: str) -> dict | None:
    """
    Get all line parameters from the given multi-index dataframe.

    Parameters
    ----------
    line_indexes : list[int]
        List of line indexes to retrieve parameters for.
    mw : object
        Main window object containing the UI elements.
    filename : str
        Filename to use for retrieving data from the dataframe.

    Returns
    -------
    dict | None
        Dictionary of parameters for each line index if found, otherwise None.
    """
    df_params = mw.ui.fit_params_table.model().get_df_by_multiindex(filename)
    parameters = {}
    for idx in line_indexes:
        line_type = mw.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Type')
        a = df_params.loc[(idx, 'a')].Value
        x0 = df_params.loc[(idx, 'x0')].Value
        dx = df_params.loc[(idx, 'dx')].Value
        result = {'a': a, 'x0': x0, 'dx': dx}
        if 'add_params' not in peak_shapes_params()[line_type]:
            parameters[idx] = result
            continue
        add_params = peak_shapes_params()[line_type]['add_params']
        for param_name in add_params:
            result[param_name] = df_params.loc[(idx, param_name)].Value
        parameters[idx] = result
    return parameters


def get_curve_for_deconvolution(n_array: np.ndarray, idx: int, style: dict) -> PlotCurveItem:
    """
    Create a PlotCurveItem for deconvolution based on provided data and style.

    Parameters
    ----------
    n_array : np.ndarray
        Numpy array containing the data points.
    idx : int
        Index of the curve.
    style : dict
        Dictionary containing the style parameters.

    Returns
    -------
    PlotCurveItem
        The created PlotCurveItem object for deconvolution.
    """
    curve = PlotCurveItem(skipFiniteCheck=True, useCache=True, clickable=True, name=idx)
    curve.setSkipFiniteCheck(True)
    curve.setClickable(True)
    pen, brush = curve_pen_brush_by_style(style)
    curve.setPen(pen)
    curve.setData(x=n_array[:, 0], y=n_array[:, 1], fillLevel=0.0, brush=brush)
    return curve


def random_line_style() -> dict:
    """
    Generate a random line style dictionary.

    Returns
    -------
    dict
        Dictionary containing random line style parameters.
    """
    return {'color': QColor().fromRgb(random_rgb()),
            'style': Qt.PenStyle.SolidLine,
            'width': 1.0,
            'fill': False,
            'use_line_color': True,
            'fill_color': QColor().fromRgb(random_rgb()),
            'fill_opacity': 0.5}


def curve_pen_brush_by_style(style: dict) -> tuple[mkPen, mkBrush]:
    """
    Create pen and brush objects based on the provided style dictionary.

    Parameters
    ----------
    style : dict
        Dictionary containing the style parameters.

    Returns
    -------
    tuple[mkPen, mkBrush]
        The created pen and brush objects.
    """
    color = style['color']
    color.setAlphaF(1.0)
    pen = mkPen(color=color, style=style['style'], width=style['width'])
    fill_color = style['color'] if style['use_line_color'] else style['fill_color']
    fill_color.setAlphaF(style['fill_opacity'])
    brush = mkBrush(fill_color) if style['fill'] else None
    return pen, brush
