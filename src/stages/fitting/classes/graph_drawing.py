import dataclasses
from os import environ

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from qtpy.QtGui import QMouseEvent, QColor
from pyqtgraph import ROI, PlotCurveItem, mkBrush, mkPen, ErrorBarItem, SignalProxy, InfiniteLine, \
    FillBetweenItem
from qtpy.QtCore import QObject, QTimer

from qfluentwidgets import MessageBox
from src import get_parent, UndoCommand
from src.data.default_values import peak_shapes_params
from src.stages import cut_full_spectrum, packed_current_line_parameters
from src.stages.fitting.functions.plotting import set_roi_size_pos, set_roi_size, \
    deconvolution_data_items_by_idx, curve_pen_brush_by_style, update_curve_style, \
    all_lines_parameters, get_curve_for_deconvolution
from src.stages.preprocessing.functions import cut_axis
from src.stages.preprocessing.functions.averaging import get_average_spectrum


@dataclasses.dataclass
class CurveSelect:
    timer_fill: QTimer | None
    curve_idx: int | None
    rad: float | None


class GraphDrawing(QObject):
    """
    parent - DecompositionStage
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.curve_select = CurveSelect(rad=None, timer_fill=None, curve_idx=None)
        self._initial_deconvolution_plot()

    def cut_data_interval(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        n_array = self.array_of_current_filename_in_deconvolution()
        if n_array is None:
            return
        n_array = cut_full_spectrum(n_array, mw.ui.interval_start_dsb.value(),
                                    mw.ui.interval_end_dsb.value())
        self.parent.curves.data.setData(x=n_array[:, 0], y=n_array[:, 1])
        self.parent.graph_drawing.assign_style('data')

    def array_of_current_filename_in_deconvolution(self) -> np.ndarray | None:
        """
        @return: 2D массив спектра, который отображается на графике в данный момент
        """
        arr = None
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        averaged_dict = context.preprocessing.stages.av_data.data
        stage = mw.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if self.parent.data.is_template:
            if mw.ui.template_combo_box.currentText() == 'Average':
                arr = self.parent.data.averaged_spectrum
            elif averaged_dict:
                arr = averaged_dict[int(mw.ui.template_combo_box.currentText().split('.')[0])]
        elif self.parent.data.current_spectrum_name in data:
            arr = data[self.parent.data.current_spectrum_name]
        else:
            return
        return arr

    def redraw_curve_by_index(self, idx: int, update: bool = True) -> None:
        mw = get_parent(self.parent, "MainWindow")
        params = self.parent.current_line_parameters(idx)
        data_items = mw.ui.deconv_plot_widget.getPlotItem().listDataItems()
        items = deconvolution_data_items_by_idx(idx, data_items)
        if items is None:
            return
        curve, roi = items
        if curve is None or roi is None:
            return
        line_type = mw.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Type')
        set_roi_size_pos((params['a'], params['x0'], params['dx']), roi, update)
        self.parent.graph_drawing.redraw_curve(params, curve, line_type, idx)

    def add_deconv_curve_to_plot(self, params: dict, idx: int, style: dict, line_type: str) \
            -> None:
        x0 = params['x0']
        dx = params['dx']
        if 'x_axis' not in params:
            params['x_axis'] = self.x_axis_for_line(x0, dx)
        x_axis = params['x_axis']
        full_amp_line, x_axis, _ = self.curve_y_x_idx(line_type, params, x_axis, idx)
        self.create_roi_curve_add_to_plot(full_amp_line, x_axis, idx, style, params)

    def update_sigma3_curves(self, filename: str | None = None) -> None:
        """
        Update self.sigma3_top and self.sigma3_bottom for current spectrum
        @param filename: optional
        @return: None
        """
        mw = get_parent(self.parent, "MainWindow")
        if filename is None:
            filename = "" if (self.parent.data.is_template
                              or self.parent.data.current_spectrum_name == '') \
                else self.parent.data.current_spectrum_name
        if not self.parent.data.sigma3:
            return
        if filename not in self.parent.data.sigma3 and self.parent.curves.sigma3_fill:
            self.parent.curves.sigma3_fill.setVisible(False)
            return
        if self.parent.curves.sigma3_fill:
            self.parent.curves.sigma3_fill.setVisible(mw.ui.sigma3_checkBox.isChecked())
        if filename not in self.parent.data.sigma3:
            return
        self.parent.curves.sigma3_top.setData(x=self.parent.data.sigma3[filename][0],
                                     y=self.parent.data.sigma3[filename][1])
        self.parent.curves.sigma3_bottom.setData(x=self.parent.data.sigma3[filename][0],
                                        y=self.parent.data.sigma3[filename][2])

    def assign_style(self, style_type: str):
        match style_type:
            case 'data':
                style = self.parent.styles.data
                curve = self.parent.curves.data
            case 'sum':
                style = self.parent.styles.sum
                curve = self.parent.curves.sum
            case 'residual':
                style = self.parent.styles.residual
                curve = self.parent.curves.residual
            case 'sigma3':
                style = self.parent.styles.sigma3
                curve = self.parent.curves.sigma3_fill
            case _:
                return
        color = style['color']
        color.setAlphaF(1.0)
        pen = mkPen(color=color, style=style['style'], width=style['width'])
        if style_type == 'sigma3' and self.parent.curves.sigma3_fill is not None:
            pen, brush = curve_pen_brush_by_style(style)
            self.parent.curves.sigma3_fill.setPen(pen)
            self.parent.curves.sigma3_fill.setBrush(brush)
        elif curve is not None:
            curve.setPen(pen)

    def x_axis_for_line(self, x0: float, dx: float) -> np.ndarray:
        mw = get_parent(self.parent, "MainWindow")
        stage = mw.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if data:
            return self.parent.get_x_axis()
        else:
            return np.array(range(int(x0 - dx * 20), int(x0 + dx * 20)))

    def curve_y_x_idx(self, line_type: str, params: dict | None, x_axis: np.ndarray = None,
                      idx: int = 0) \
            -> tuple[np.ndarray, np.ndarray, int]:
        """Returns y-axis and x_axis of deconvolution line

        Inputs
        ------
        line_type : str
            name of line from self.deconv_line_params.keys()
        params : dict
            with parameters value - a, x0, dx and etc.
        x_axis : np.ndarray
            corresponding x_axis of y_axis line

        Returns
        -------
        out : tuple[np.ndarray, np.ndarray]
            y, x axis signal
        """
        mw = get_parent(self.parent, "MainWindow")
        dx, x0 = params['dx'], params['x0']
        if x_axis is None:
            x_axis = params.pop('x_axis') if 'x_axis' in params \
                else self.x_axis_for_line(x0, dx)

        func = peak_shapes_params()[line_type]['func']
        func_param = []
        if 'add_params' in peak_shapes_params()[line_type]:
            add_params = peak_shapes_params()[line_type]['add_params']
            for i in add_params:
                if i in params:
                    func_param.append(params[i])
        if 'x_axis' in params:
            params.pop('x_axis')
        params = {k: v for k, v in params.items() if k in func.__annotations__.keys()}
        y = func(x_axis, **params)

        if mw.ui.interval_checkBox.isChecked() and y is not None:
            x_axis, idx_start, idx_end = cut_axis(x_axis, mw.ui.interval_start_dsb.value(),
                                                  mw.ui.interval_end_dsb.value())
            y = y[idx_start: idx_end + 1]
        return y, x_axis, idx

    # region draw
    def draw_sum_curve(self) -> None:
        """
            Update sum curve

            Returns
            -------
            out : None
        """
        if self.parent.curves.sum is None:
            return
        x_axis, y_axis = self.parent.sum_array()
        self.parent.curves.sum.setData(x=x_axis, y=y_axis.T)
        self.assign_style('sum')

    def draw_residual_curve(self) -> None:
        """
            Update residual curve after sum curve updated.
            Residual = data - sum

            Returns
            -------
            out : None
        """
        if self.parent.curves.residual is None:
            return
        x_axis, y_axis = self.residual_array()
        self.parent.curves.residual.setData(x=x_axis, y=y_axis.T)
        self.assign_style('residual')

    def redraw_curve(self, params: dict | None = None, curve: PlotCurveItem = None,
                     line_type: str | None = None,
                     idx: int | None = None) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if params is None and idx is not None:
            params = self.parent.current_line_parameters(idx)
        elif params is None:
            return
        if curve is None and idx is not None:
            data_items = mw.ui.deconv_plot_widget.getPlotItem().listDataItems()
            curve = deconvolution_data_items_by_idx(idx, data_items)[0]
        elif curve is None:
            return
        if line_type is None and idx is not None:
            line_type = mw.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Type')
        elif line_type is None:
            return
        x0 = params['x0']
        full_amp_line, x_axis, _ = self.curve_y_x_idx(line_type, params, idx=idx)
        if full_amp_line is None:
            return
        curve.setData(x=x_axis, y=full_amp_line)
        curve.setPos(-x0, 0)

    def draw_all_curves(self) -> None:
        """
        Отрисовка всех линий
        @return: None
        """
        mw = get_parent(self.parent, "MainWindow")
        line_indexes = mw.ui.deconv_lines_table.model().dataframe().index
        model = mw.ui.deconv_lines_table.model()
        filename = "" if (self.parent.data.is_template
                          or self.parent.data.current_spectrum_name == '') \
            else self.parent.data.current_spectrum_name
        cur_all_lines_params = all_lines_parameters(line_indexes, mw, filename)
        result = []
        for i in line_indexes:
            res = self.curve_y_x_idx(model.cell_data_by_idx_col_name(i, 'Type'),
                                     cur_all_lines_params[i], idx=i)
            result.append(res)

        for full_amp_line, x_axis, idx in result:
            self.create_roi_curve_add_to_plot(full_amp_line, x_axis, idx,
                                              model.cell_data_by_idx_col_name(idx, 'Style'),
                                              cur_all_lines_params[idx])
        self.draw_sum_curve()
        self.draw_residual_curve()

    def create_roi_curve_add_to_plot(self, full_amp_line: np.ndarray | None,
                                     x_axis: np.ndarray | None, idx: int,
                                     style: dict, params: dict) -> None:
        mw = get_parent(self.parent, "MainWindow")
        a, x0, dx = params['a'], params['x0'], params['dx']
        if full_amp_line is None:
            return
        n_array = np.vstack((x_axis, full_amp_line)).T
        curve = get_curve_for_deconvolution(n_array, idx, style)
        curve.sigClicked.connect(self.curve_clicked)
        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        plot_item.addItem(curve)
        roi = ROI([x0, 0], [dx, a], resizable=False, removable=True, rotatable=False, movable=False,
                  pen='transparent')
        roi.addTranslateHandle([0, 1])
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        plot_item.addItem(roi)
        curve.setParentItem(roi)
        curve.setPos(-x0, 0)
        if not mw.ui.deconv_lines_table.model().checked()[idx] and roi is not None:
            roi.setVisible(False)
        roi.sigRegionChangeStarted.connect(
            lambda checked=None, index=idx: self.curve_roi_pos_change_started(index,
                                                                              roi))
        roi.sigRegionChangeFinished.connect(
            lambda checked=None, index=idx: self.curve_roi_pos_change_finished(index,
                                                                                             roi))
        roi.sigRegionChanged.connect(
            lambda checked=None, index=idx: self.curve_roi_pos_changed(index, roi,
                                                                                     curve))

    def curve_roi_pos_change_started(self, index: int, roi: ROI) -> None:
        params = self.parent.current_line_parameters(index)
        if not params:
            return
        a, x0, dx = params['a'], params['x0'], params['dx']
        color = QColor(environ['secondaryColor'])
        color.setAlphaF(0.5)
        roi.setPen(color)
        self.parent.plotting.dragged_line_parameters = a, x0, dx

    def redraw_curves_for_filename(self) -> None:
        """
        Redraw all curves by parameters of current selected spectrum.
        Also add ErrorBarItem.
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        fn = "" if self.parent.data.is_template or self.parent.data.current_spectrum_name == '' \
            else self.parent.data.current_spectrum_name
        line_indexes = mw.ui.deconv_lines_table.model().get_visible_line_types().index
        filename_lines_indexes = mw.ui.fit_params_table.model().get_lines_indexes_by_filename(fn)
        if filename_lines_indexes is None:
            return
        line_types = mw.ui.deconv_lines_table.model().column_data(1)
        if line_types.empty:
            return
        if len(line_indexes) != len(filename_lines_indexes):
            MessageBox("Template error",
                       "The composition of the lines in this spectrum differs from the template."
                       " Some lines will not be drawn correctly.", mw, {'Ok'}).exec()
            line_indexes = filename_lines_indexes
        params = self.current_filename_lines_parameters(list(line_indexes), fn, line_types)
        if params is None or not params:
            return
        if len(line_indexes) == 0:
            return
        items = {i.name(): (i, i.parentItem()) for i in plot_item.listDataItems()
                 if i.name() in list(line_indexes)}
        if items is None or not items:
            return
        create_error_bars = (fn in self.parent.data.params_stderr
                             and self.parent.data.params_stderr[fn])
        error_bar_params = {'x': [], 'y': [], 'h': [], 'w': []}
        # Remove old ErrorBarItem.
        for i in plot_item.items:
            if isinstance(i, ErrorBarItem):
                plot_item.removeItem(i)

        for i in line_indexes:
            if i not in params or i not in items:
                continue
            set_roi_size_pos((params[i]['a'], params[i]['x0'], params[i]['dx']), items[i][1], False)
            self.redraw_curve(params[i], items[i][0], line_types.loc[i], i)
            # Errors bars using stderr.
            if not create_error_bars:
                continue
            error_bar_params['x'].extend([params[i]['x0'], params[i]['x0'] + params[i]['dx']])
            error_bar_params['y'].extend([params[i]['a'], params[i]['a'] / 2])
            try:
                stderr = self.parent.data.params_stderr[fn][i]
            except KeyError:
                create_error_bars = False
                continue
            error_bar_params['h'].extend([stderr['a'], 0.])
            error_bar_params['w'].extend([stderr['x0'], stderr['dx']])
        if not create_error_bars:
            return
        eb = ErrorBarItem(x=np.array(error_bar_params['x']), y=np.array(error_bar_params['y']),
                          height=2 * np.array(error_bar_params['h']),
                          width=2 * np.array(error_bar_params['w']), x_beam=1, y_beam=.005,
                          pen=mkPen(color='black', width=2))
        plot_item.addItem(eb)

    def current_filename_lines_parameters(self, indexes: list[int], filename,
                                          line_types: pd.Series) -> dict | None:
        mw = get_parent(self.parent, "MainWindow")
        df_params = mw.ui.fit_params_table.model().get_df_by_multiindex(filename)
        if df_params.empty:
            return None
        params = {}
        for idx in indexes:
            params[idx] = packed_current_line_parameters(df_params.loc[idx], line_types.loc[idx],
                                                         peak_shapes_params())
        return params

    def update_single_deconvolution_plot(self, current_spectrum_name: str,
                                         is_template: bool = False,
                                         is_averaged_or_group: bool = False) -> None:
        """
        Change current data spectrum in deconv_plot_widget
        set self.isTemplate

        isTemplate - True if current spectrum is averaged or group's averaged
        isAveraged_or_Group
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        stage = mw.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if not data:
            return
        self.parent.data.is_template = is_template
        arr = None
        if is_template and is_averaged_or_group:
            self.parent.data.current_spectrum_name = ''
            averaged_dict = context.preprocessing.stages.av_data.data
            if current_spectrum_name == 'Average':
                av_method = context.preprocessing.stages.av_data.ui.average_method_cb.currentText()
                arr = get_average_spectrum(list(data.values()), av_method)
                self.parent.data.averaged_spectrum = arr
                mw.ui.max_noise_level_dsb.setValue(np.max(arr[:, 1]) / 99.)
            elif averaged_dict:
                arr = averaged_dict[int(current_spectrum_name.split('.')[0])]
        else:
            arr = data[current_spectrum_name]
            self.parent.data.current_spectrum_name = current_spectrum_name
        if arr is None:
            return
        if mw.ui.interval_checkBox.isChecked():
            arr = cut_full_spectrum(arr, mw.ui.interval_start_dsb.value(),
                                    mw.ui.interval_end_dsb.value())
        title_text = 'Template. ' + current_spectrum_name if is_template else current_spectrum_name
        new_title = ("<span style=\"font-family: AbletonSans; color:" + environ['plotText']
                     + ";font-size:14pt\">" + title_text + "</span>")

        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        if self.parent.curves.data:
            plot_item.removeItem(self.parent.curves.data)
        if self.parent.curves.sum:
            plot_item.removeItem(self.parent.curves.sum)
        if self.parent.curves.residual:
            plot_item.removeItem(self.parent.curves.residual)
        self.parent.curves.data = mw.get_curve_plot_data_item(arr,
                                                              color=self.parent.styles.data[
                                                                  'color'], name='data')
        self.parent.curves.sum = mw.get_curve_plot_data_item(
            np.vstack((self.parent.sum_array())).T,
            color=self.parent.styles.sum['color'], name='sum')
        self.parent.curves.residual = mw.get_curve_plot_data_item(
            np.vstack((self.residual_array())).T,
            color=self.parent.styles.residual['color'], name='residual')
        self.parent.curves.data.setVisible(mw.ui.data_checkBox.isChecked())
        self.parent.curves.sum.setVisible(mw.ui.sum_checkBox.isChecked())
        self.parent.curves.residual.setVisible(mw.ui.residual_checkBox.isChecked())
        self.assign_style('data')
        self.assign_style('sum')
        self.assign_style('residual')
        self.assign_style('sigma3')
        plot_item.addItem(self.parent.curves.data, kargs=['ignoreBounds', 'skipAverage'])
        plot_item.addItem(self.parent.curves.sum, kargs=['ignoreBounds', 'skipAverage'])
        plot_item.addItem(self.parent.curves.residual, kargs=['ignoreBounds', 'skipAverage'])
        plot_item.getViewBox().updateAutoRange()
        mw.ui.deconv_plot_widget.setTitle(new_title)

    def residual_array(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Return x, y arrays of residual spectra.
            Residual = data - sum

            Returns
            -------
            out : tuple[np.ndarray, np.ndarray]
                x_axis, y_axis of residual curve
        """
        x_data, y_data = self.parent.curves.data.getData()
        _, y_sum = self.parent.curves.sum.getData()
        c = y_data.copy()
        c[:len(y_sum)] -= y_sum
        return x_data, c

    # endregion

    # region curve_select
    def curve_roi_pos_change_finished(self, index: int, roi: ROI) -> None:
        params = self.parent.current_line_parameters(index)
        if not params:
            return
        a, x0, dx = params['a'], params['x0'], params['dx']
        roi_a, roi_x0, roi_dx = roi.size().y(), roi.pos().x(), roi.size().x()

        if (a, x0, dx) != self.parent.plotting.dragged_line_parameters and (
                a, x0, dx) != self.parent.plotting.prev_dragged_line_parameters and \
                (a, x0, dx) != (roi_a, roi_x0, roi_dx):
            context = get_parent(self.parent, "Context")
            command = CommandDeconvLineDragged((a, x0, dx,
                                                self.parent.plotting.dragged_line_parameters, roi),
                                               context, text=f"Edit line {index}")
            roi.setPen('transparent')
            context.undo_stack.push(command)
            self.parent.plotting.prev_dragged_line_parameters = (
                self.parent.plotting.dragged_line_parameters)
            self.parent.plotting.dragged_line_parameters = a, x0, dx

    def curve_roi_pos_changed(self, index: int, roi: ROI, curve: PlotCurveItem) -> None:
        dx, x0 = roi.size().x(), roi.pos().x()
        new_height = roi.pos().y() + roi.size().y()
        params = self.parent.current_line_parameters(index)
        if not params:
            return
        mw = get_parent(self.parent, "MainWindow")
        model = mw.ui.fit_params_table.model()
        filename = '' if self.parent.data.is_template else self.parent.data.current_spectrum_name
        line_type = mw.ui.deconv_lines_table.model().cell_data_by_idx_col_name(index, 'Type')
        if x0 < params['min_x0']:
            model.set_parameter_value(filename, index, 'x0', 'Min value', x0 - 1)
        if x0 > params['max_x0']:
            model.set_parameter_value(filename, index, 'x0', 'Max value', x0 + 1)
        if new_height < params['min_a']:
            model.set_parameter_value(filename, index, 'a', 'Min value', new_height)
        if new_height > params['max_a']:
            model.set_parameter_value(filename, index, 'a', 'Max value', new_height)
        if dx < params['min_dx']:
            model.set_parameter_value(filename, index, 'dx', 'Min value', dx - dx / 2)
        if dx > params['max_dx']:
            model.set_parameter_value(filename, index, 'dx', 'Max value', dx + dx / 2)
        if np.round(new_height, 5) == np.round(params['a'], 5) and np.round(dx, 5) == np.round(
                params['dx'], 5) and np.round(params['x0'], 5) == np.round(x0, 5):
            return
        model.set_parameter_value(filename, index, 'dx', 'Value', dx)
        model.set_parameter_value(filename, index, 'a', 'Value', new_height)
        model.set_parameter_value(filename, index, 'x0', 'Value', x0)
        set_roi_size(roi.size().x(), new_height, roi)
        params = {'a': new_height, 'x0': x0, 'dx': dx}
        if 'add_params' not in peak_shapes_params()[line_type]:
            self.parent.graph_drawing.redraw_curve(params, curve, line_type)
            return
        add_params = peak_shapes_params()[line_type]['add_params']
        for param_name in add_params:
            params[param_name] = model.get_parameter_value(filename, index, param_name, 'Value')
        self.parent.graph_drawing.redraw_curve(params, curve, line_type)

    def start_fill_timer(self, idx: int) -> None:
        mw = get_parent(self.parent, "MainWindow")
        self.curve_select.rad = 0.
        self.curve_select.curve_idx = idx
        self.curve_select.timer_fill = QTimer(mw)
        self.curve_select.timer_fill.timeout.connect(self.update_curve_fill_realtime)
        self.curve_select.timer_fill.start(10)

    def update_curve_fill_realtime(self):
        self.curve_select.rad += 0.02
        idx = self.curve_select.curve_idx
        mw = get_parent(self.parent, "MainWindow")
        if idx not in list(mw.ui.deconv_lines_table.model().indexes):
            self.deselect_selected_line()
            return
        sin_v = np.abs(np.sin(self.curve_select.rad))
        if mw.ui.deconv_lines_table.model().rowCount() == 0:
            return
        curve_style = mw.ui.deconv_lines_table.model().cell_data_by_idx_col_name(idx, 'Style')
        fill_color = curve_style['color'] if curve_style['use_line_color'] else curve_style[
            'fill_color']
        fill_color.setAlphaF(sin_v)
        brush = mkBrush(fill_color)
        data_items = mw.ui.deconv_plot_widget.getPlotItem().listDataItems()
        curve, _ = deconvolution_data_items_by_idx(idx, data_items)
        if not curve:
            self.deselect_selected_line()
            return
        curve.setBrush(brush)

    def select_curve(self, idx: int) -> None:
        mw = get_parent(self.parent, "MainWindow")
        row = mw.ui.deconv_lines_table.model().row_by_index(idx)
        mw.ui.deconv_lines_table.selectRow(row)
        self.parent.set_rows_visibility()
        self.start_fill_timer(idx)

    def deselect_selected_line(self) -> None:
        if self.curve_select.timer_fill is None:
            return
        self.curve_select.timer_fill.stop()
        mw = get_parent(self.parent, "MainWindow")
        if (self.curve_select.curve_idx is not None
                and self.curve_select.curve_idx
                in list(mw.ui.deconv_lines_table.model().indexes)):
            curve_style = mw.ui.deconv_lines_table.model().cell_data_by_idx_col_name(
                self.curve_select.curve_idx, 'Style')
            data_items = mw.ui.deconv_plot_widget.getPlotItem().listDataItems()
            update_curve_style(self.curve_select.curve_idx, curve_style, data_items)
            self.curve_select.curve_idx = None

    def curve_clicked(self, curve: PlotCurveItem, _event: QMouseEvent) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if (self.curve_select.curve_idx is not None \
                and self.curve_select.curve_idx
                in mw.ui.deconv_lines_table.model().dataframe().index):
            curve_style = mw.ui.deconv_lines_table.model().cell_data_by_idx_col_name(
                self.curve_select.curve_idx, 'Style')
            data_items = mw.ui.deconv_plot_widget.getPlotItem().listDataItems()
            update_curve_style(self.curve_select.curve_idx, curve_style, data_items)
        idx = int(curve.name())
        self.select_curve(idx)
    # endregion

    def _initial_deconvolution_plot(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        mw.ui.deconv_plot_widget.scene().sigMouseClicked.connect(self.fit_plot_mouse_clicked)
        mw.crosshair_update_deconv = SignalProxy(
            mw.ui.deconv_plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=mw.update_crosshair_deconv_plot,
        )
        mw.ui.deconv_plot_widget.setAntialiasing(1)
        plot_item.enableAutoRange()
        plot_item.showGrid(True, True, 0.5)
        mw.ui.deconv_plot_widget.vertical_line = InfiniteLine()
        mw.ui.deconv_plot_widget.horizontal_line = InfiniteLine(angle=0)
        mw.ui.deconv_plot_widget.vertical_line.setPen(QColor(environ["secondaryColor"]))
        mw.ui.deconv_plot_widget.horizontal_line.setPen(QColor(environ["secondaryColor"]))
        items_matches = (x for x in plot_item.listDataItems() if not x.name())
        for i in items_matches:
            plot_item.removeItem(i)
        self.initial_deconv_plot_color()
        plot_item.addItem(self.parent.plotting.linear_region)
        self.parent.curves.sigma3_top = PlotCurveItem(name="sigma3_top")
        self.parent.curves.sigma3_bottom = PlotCurveItem(name="sigma3_bottom")
        color = self.parent.styles.sigma3["color"]
        color.setAlphaF(0.25)
        pen = mkPen(color=color, style=Qt.PenStyle.SolidLine)
        brush = mkBrush(color)
        self.parent.curves.sigma3_fill = FillBetweenItem(
            self.parent.curves.sigma3_top,
            self.parent.curves.sigma3_bottom, brush, pen
        )
        plot_item.addItem(self.parent.curves.sigma3_fill)
        self.parent.curves.sigma3_fill.setVisible(False)

    def fit_plot_mouse_clicked(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.deselect_selected_line()

    def initial_deconv_plot_color(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.deconv_plot_widget.setBackground(mw.plot_background_color)
        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        plot_item.getAxis("bottom").setPen(mw.plot_text_color)
        plot_item.getAxis("left").setPen(mw.plot_text_color)
        plot_item.getAxis("bottom").setTextPen(mw.plot_text_color)
        plot_item.getAxis("left").setTextPen(mw.plot_text_color)


class CommandDeconvLineDragged(UndoCommand):
    """
    UNDO/REDO change position of line ROI

    Parameters
    -------
    data: tuple[float, float, float, tuple, ROI]
        'a', 'x0', 'dx', dragged_line_parameters, ROI
    parent: Context
        Backend context class
    text: str
        description
    """

    def __init__(self, data: tuple[float, float, float, tuple[float, float, float], ROI],
                 parent, text: str, *args, **kwargs) -> None:
        super().__init__(data, parent, text, *args, **kwargs)
        self._params = data[0], data[1], data[2]
        self._old_params = data[3]
        self._roi = data[4]

    def redo_special(self):
        """
        Update data and input table columns
        """
        set_roi_size_pos(self._params, self._roi)

    def undo_special(self):
        """
        Undo data and input table columns
        """
        set_roi_size_pos(self._old_params, self._roi)

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        d = self.parent.decomposition
        d.graph_drawing.draw_sum_curve()
        d.graph_drawing.draw_residual_curve()
        self.parent.set_modified()
