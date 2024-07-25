import numpy as np
from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QColor
from pyqtgraph import ROI, PlotCurveItem, mkPen, mkBrush

from src.data.default_values import peak_shapes_params
from src.data.plotting import random_rgb


def set_roi_size_pos(params: tuple[float, float, float], roi: ROI, update: bool = True) -> None:
    a, x0, dx = params
    set_roi_size(dx, a, roi, update=update)
    set_roi_pos(x0, roi)


def set_roi_size(x: float, y: float, roi: ROI, center=None, finish: bool = False,
                 update: bool = True) -> None:
    if center is None:
        center = [0, 1]
    new_size = QPointF()
    new_size.setX(x)
    new_size.setY(y)
    roi.setSize(new_size, center, finish=finish, update=update)


def set_roi_pos(x0: float, roi: ROI) -> None:
    new_pos = QPointF()
    new_pos.setX(x0)
    new_pos.setY(0.0)
    roi.setPos(new_pos)


def deconvolution_data_items_by_idx(idx: int, data_items: list) -> tuple[PlotCurveItem, ROI] | None:
    if len(data_items) == 0:
        return
    curve = None
    roi = None
    for i in data_items:
        if i.name() == idx:
            curve = i
            roi = i.parentItem()
            break
    return curve, roi

def update_curve_style(idx: int, style: dict, data_items: list) -> None:
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
    curve = PlotCurveItem(skipFiniteCheck=True, useCache=True, clickable=True, name=idx)
    curve.setSkipFiniteCheck(True)
    curve.setClickable(True)
    pen, brush = curve_pen_brush_by_style(style)
    curve.setPen(pen)
    curve.setData(x=n_array[:, 0], y=n_array[:, 1], fillLevel=0.0, brush=brush)
    return curve

def random_line_style() -> dict:
    return {'color': QColor().fromRgb(random_rgb()),
            'style': Qt.PenStyle.SolidLine,
            'width': 1.0,
            'fill': False,
            'use_line_color': True,
            'fill_color': QColor().fromRgb(random_rgb()),
            'fill_opacity': 0.5}


def curve_pen_brush_by_style(style: dict) -> tuple[mkPen, mkBrush]:
    color = style['color']
    color.setAlphaF(1.0)
    pen = mkPen(color=color, style=style['style'], width=style['width'])
    fill_color = style['color'] if style['use_line_color'] else style['fill_color']
    fill_color.setAlphaF(style['fill_opacity'])
    brush = mkBrush(fill_color) if style['fill'] else None
    return pen, brush
