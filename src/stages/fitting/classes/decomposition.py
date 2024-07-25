from gc import get_objects
from os import environ

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QMenu
from lmfit.model import ModelResult
from pandas import MultiIndex, DataFrame
from qtpy.QtCore import QObject, Qt
from qtpy.QtGui import QColor, QMouseEvent
from pyqtgraph import PlotCurveItem, LinearRegionItem

from qfluentwidgets import MessageBox
from src import get_parent, get_config
from src.data.collections import NestedDefaultDict
from src.data.default_values import peak_shapes_params
from src.data.plotting import random_rgb
from src.data.work_with_arrays import nearest_idx, find_nearest
from src.stages import packed_current_line_parameters, curve_idx_from_par_name
from src.stages.preprocessing.functions import cut_axis
from src.widgets import CurvePropertiesWindow
from .dataclasses import Curves, Data, Tables, Plotting, Styles
from .decomposition_backend import DecompositionBackend
from .graph_drawing import GraphDrawing
from .table_fit_borders import TableFitBorders
from .table_filenames import TableFilenames
from .table_decomp_lines import TableDecompLines
from .table_params import TableParams
from .undo import CommandAddDeconvLine, CommandUpdateDataCurveStyle


# pylint: disable=too-many-instance-attributes
# 8 is reasonable

class DecompositionStage(QObject):
    """
    params_stderr:
        key - filename; value - dict
                        with key - curve_idx and value - dict
                                                with key - param_name; value - stderr.
        Example  {'filename1': {0: {'a': 0.1, 'x0': 1.1, 'dx': .5},...}, ...}
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.graph_drawing = None
        self.b = None
        self.plotting = None
        self.tables = None
        self.data = None
        self.curves = None
        self.styles = None
        self._set_ui()
        self.reset()

    def _set_styles(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        data_style = {"color": QColor(environ["secondaryColor"]), "style": Qt.PenStyle.SolidLine,
                      "width": 1.0, "fill": False, "use_line_color": True,
                      "fill_color": QColor().fromRgb(random_rgb()), "fill_opacity": 0.0}
        sum_style = {"color": QColor(environ["primaryColor"]), "style": Qt.PenStyle.DashLine,
                     "width": 1.0, "fill": False, "use_line_color": True,
                     "fill_color": QColor().fromRgb(random_rgb()), "fill_opacity": 0.0}
        residual_style = {
            "color": QColor(environ["secondaryLightColor"]), "style": Qt.PenStyle.DotLine,
            "width": 1.0, "fill": False, "use_line_color": True,
            "fill_color": QColor().fromRgb(random_rgb()), "fill_opacity": 0.0}
        sigma3_style = {
            "color": QColor(environ["primaryDarkColor"]), "style": Qt.PenStyle.SolidLine,
            "width": 1.0, "fill": True, "use_line_color": True,
            "fill_color": QColor().fromRgb(random_rgb()), "fill_opacity": 0.25}
        mw.ui.data_pushButton.setStyleSheet(
            f"""*{{background-color: {data_style["color"].name()};}}""")
        mw.ui.sum_pushButton.setStyleSheet(
            f"""*{{background-color: {sum_style["color"].name()};}}""")
        mw.ui.residual_pushButton.setStyleSheet(
            f"""*{{background-color: {residual_style["color"].name()};}}""")
        mw.ui.sigma3_pushButton.setStyleSheet(
            f"""*{{background-color: {sigma3_style["color"].name()};}}""")
        self.styles = Styles(data=data_style, sum=sum_style, residual=residual_style,
                             sigma3=sigma3_style)

    def _set_ui(self):
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        fitting_methods = get_config("fitting")['fitting_methods']
        mw.ui.template_combo_box.currentTextChanged.connect(self.switch_template)
        mw.ui.interval_checkBox.stateChanged.connect(self._interval_cb_state_changed)
        mw.ui.interval_start_dsb.mouseDoubleClickEvent = lambda event: (
            self._reset_field(event, 'interval_start'))
        mw.ui.interval_start_dsb.valueChanged.connect(self._interval_start_changed)
        mw.ui.interval_end_dsb.setMaximum(99_999.0)
        mw.ui.interval_end_dsb.valueChanged.connect(self._interval_end_changed)
        mw.ui.fit_opt_method_comboBox.clear()
        mw.ui.fit_opt_method_comboBox.addItems(fitting_methods)
        mw.ui.fit_opt_method_comboBox.currentTextChanged.connect(context.set_modified)
        mw.ui.guess_method_cb.clear()
        mw.ui.guess_method_cb.addItems(["Average", "Average groups", "All"])
        mw.ui.guess_method_cb.currentTextChanged.connect(context.set_modified)
        mw.ui.max_dx_dsb.valueChanged.connect(context.set_modified)
        mw.ui.max_noise_level_dsb.valueChanged.connect(context.set_modified)
        mw.ui.l_ratio_doubleSpinBox.valueChanged.connect(context.set_modified)
        mw.ui.include_x0_checkBox.stateChanged.connect(self._set_deconvoluted_dataset)
        self._initial_add_line_button()
        mw.ui.fit_pushButton.clicked.connect(lambda: self.b.fit())
        self._initial_guess_button()
        mw.ui.batch_button.clicked.connect(lambda: self.b.batch_fit())
        mw.ui.data_checkBox.stateChanged.connect(self._data_cb_state_changed)
        mw.ui.sum_checkBox.stateChanged.connect(self._sum_cb_state_changed)
        mw.ui.sigma3_checkBox.stateChanged.connect(self._sigma3_cb_state_changed)
        mw.ui.residual_checkBox.stateChanged.connect(self._residual_cb_state_changed)
        mw.ui.data_pushButton.clicked.connect(lambda: self._style_pb_clicked('data'))
        mw.ui.sum_pushButton.clicked.connect(lambda: self._style_pb_clicked('sum'))
        mw.ui.residual_pushButton.clicked.connect(lambda: self._style_pb_clicked('residual'))
        mw.ui.sigma3_pushButton.clicked.connect(lambda: self._style_pb_clicked('sigma3'))

    def reset(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        defaults = get_config('defaults')
        mw.ui.deconv_plot_widget.getPlotItem().clear()
        mw.ui.current_filename_combobox.clear()
        self.update_template_combo_box()
        mw.ui.interval_checkBox.setChecked(False)
        mw.ui.interval_start_dsb.setValue(defaults['interval_start'])
        mw.ui.interval_end_dsb.setValue(defaults["interval_end"])
        mw.ui.fit_opt_method_comboBox.setCurrentText(defaults["fit_method"])
        mw.ui.guess_method_cb.setCurrentText(defaults["guess_method_cb"])
        mw.ui.max_dx_dsb.setValue(defaults["max_dx_guess"])
        mw.ui.max_noise_level_dsb.setValue(defaults["max_noise_level"])
        mw.ui.l_ratio_doubleSpinBox.setValue(defaults["l_ratio"])
        mw.ui.include_x0_checkBox.setChecked(False)
        mw.ui.use_area_check_box.setChecked(False)
        mw.ui.data_checkBox.setChecked(True)
        mw.ui.sum_checkBox.setChecked(True)
        mw.ui.report_text_edit.setText("")
        mw.ui.residual_checkBox.setChecked(True)
        mw.ui.sigma3_checkBox.setChecked(True)
        self.curves = Curves(data=None, sum=None, residual=None, sigma3_fill=None,
                             sigma3_top=PlotCurveItem(name='sigma3_top'),
                             sigma3_bottom=PlotCurveItem(name='sigma3_bottom'))
        self.data = Data(report_result={}, is_template=False, current_spectrum_name='', sigma3={},
                         averaged_spectrum=np.array([]), params_stderr=NestedDefaultDict(),
                         all_ranges_clustered_x0_sd=None)
        self.tables = Tables(fit_borders=TableFitBorders(self), filenames=TableFilenames(self),
                             decomp_lines=TableDecompLines(self),
                             fit_params_table=TableParams(self))
        self.b = DecompositionBackend(self)
        self.plotting = Plotting(dragged_line_parameters=None, prev_dragged_line_parameters=None,
                                 intervals_data=None, linear_region=None)
        self._set_styles()
        self._init_linear_region()
        self.graph_drawing = GraphDrawing(self)
        self.plotting.linear_region.setVisible(False)
        mw.ui.fit_borders_TableView.model().clear_dataframe()
        mw.ui.deconv_lines_table.model().clear_dataframe()
        mw.ui.fit_params_table.model().clear_dataframe()
        if mw.ui.deconvoluted_dataset_table_view.model() is not None:
            mw.ui.deconvoluted_dataset_table_view.model().clear_dataframe()
        if mw.ui.ignore_dataset_table_view.model() is not None:
            mw.ui.ignore_dataset_table_view.model().clear_dataframe()
        mw.ui.intervals_gb.setChecked(True)


    def _reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Change field value to default on double click by MiddleButton.

        Parameters
        -------
        event: QMouseEvent

        field_id: str
            name of field
        """
        if event.buttons() != Qt.MouseButton.MiddleButton:
            return
        mw = get_parent(self.parent, "MainWindow")
        value = get_config('defaults')[field_id]
        match field_id:
            case 'interval_start':
                mw.ui.interval_start_dsb.setValue(value)
            case 'interval_end':
                mw.ui.interval_end_dsb.setValue(value)
            case 'max_dx_guess':
                mw.ui.max_dx_dsb.setValue(value)
            case 'max_noise_level':
                mw.ui.max_noise_level_dsb.setValue(value)
            case 'l_ratio':
                mw.ui.l_ratio_doubleSpinBox.setValue(value)
            case _:
                return

    def read(self) -> dict:
        """
        Read attributes data.

        Returns
        -------
        dt: dict
            all class attributes data
        """
        mw = get_parent(self.parent, "MainWindow")
        dt = {"interval_checkBox_checked": mw.ui.interval_checkBox.isChecked(),
              'interval_start_cm': mw.ui.interval_start_dsb.value(),
              'interval_end_cm': mw.ui.interval_end_dsb.value(),
              'fit_method': mw.ui.fit_opt_method_comboBox.currentText(),
              'guess_method_cb': mw.ui.guess_method_cb.currentText(),
              'max_dx_guess': mw.ui.max_dx_dsb.value(),
              'max_noise_level': mw.ui.max_noise_level_dsb.value(),
              'l_ratio_doubleSpinBox': mw.ui.l_ratio_doubleSpinBox.value(),
              'include_x0_checkBox': mw.ui.include_x0_checkBox.isChecked(),
              'use_fit_intervals': mw.ui.intervals_gb.isChecked(),
              'intervals_table_df': mw.ui.fit_borders_TableView.model().dataframe(),
              'DeconvLinesTableDF': mw.ui.deconv_lines_table.model().dataframe(),
              'DeconvLinesTableChecked': mw.ui.deconv_lines_table.model().checked(),
              'DeconvParamsTableDF': mw.ui.fit_params_table.model().dataframe(),
              'use_area_check_box': mw.ui.use_area_check_box.isChecked(),
              'data_curve_checked': mw.ui.data_checkBox.isChecked(),
              'sum_curve_checked': mw.ui.sum_checkBox.isChecked(),
              'residual_curve_checked': mw.ui.residual_checkBox.isChecked(),
              'sigma3_checked': mw.ui.sigma3_checkBox.isChecked(),
              'plotting_intervals_data': self.plotting.intervals_data,
              'plotting_prev_dragged_line_parameters': self.plotting.prev_dragged_line_parameters,
              'plotting_dragged_line_parameters': self.plotting.dragged_line_parameters,
              'data_report_result': self.data.report_result,
              'data_current_spectrum_name': self.data.current_spectrum_name,
              'data_is_template': self.data.is_template,
              'data_sigma3': self.data.sigma3,
              'data_averaged_spectrum': self.data.averaged_spectrum,
              'data_params_stderr': self.data.params_stderr,
              'data_all_ranges_clustered_x0_sd': self.data.all_ranges_clustered_x0_sd,
              'styles': self.styles
              }
        return dt

    def load(self, db: dict) -> None:
        """
        Load attributes data from file.

        Parameters
        -------
        db: dict
            all class attributes data
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.interval_checkBox.setChecked(db["interval_checkBox_checked"])
        mw.ui.interval_start_dsb.setValue(db["interval_start_cm"])
        mw.ui.interval_end_dsb.setValue(db["interval_end_cm"])
        mw.ui.fit_opt_method_comboBox.setCurrentText(db["fit_method"])
        mw.ui.guess_method_cb.setCurrentText(db["guess_method_cb"])
        mw.ui.max_dx_dsb.setValue(db["max_dx_guess"])
        mw.ui.max_noise_level_dsb.setValue(db["max_noise_level"])
        mw.ui.l_ratio_doubleSpinBox.setValue(db["l_ratio_doubleSpinBox"])
        mw.ui.include_x0_checkBox.setChecked(db["include_x0_checkBox"])
        mw.ui.intervals_gb.setChecked(db["use_fit_intervals"])
        mw.ui.fit_borders_TableView.model().set_dataframe(db["intervals_table_df"])
        mw.ui.deconv_lines_table.model().set_dataframe(db["DeconvLinesTableDF"])
        mw.ui.deconv_lines_table.model().set_checked(db["DeconvLinesTableChecked"])
        mw.ui.fit_params_table.model().set_dataframe(db["DeconvParamsTableDF"])
        mw.ui.use_area_check_box.setChecked(db["use_area_check_box"])
        mw.ui.data_checkBox.setChecked(db["data_curve_checked"])
        mw.ui.sum_checkBox.setChecked(db["sum_curve_checked"])
        mw.ui.residual_checkBox.setChecked(db["residual_curve_checked"])
        mw.ui.sigma3_checkBox.setChecked(db["sigma3_checked"])
        self.plotting.intervals_data = db["plotting_intervals_data"]
        self.plotting.prev_dragged_line_parameters = db["plotting_prev_dragged_line_parameters"]
        self.plotting.dragged_line_parameters = db["plotting_dragged_line_parameters"]
        self.data.report_result = db['data_report_result']
        self.data.current_spectrum_name = db['data_current_spectrum_name']
        self.data.is_template = db['data_is_template']
        self.data.sigma3 = db['data_sigma3']
        self.data.averaged_spectrum = db['data_sigma3']
        self.data.params_stderr = db['data_params_stderr']
        self.data.all_ranges_clustered_x0_sd = db['data_all_ranges_clustered_x0_sd']
        self.styles = db["styles"]
        self.set_pen(self.styles.data, 'data')
        self.set_pen(self.styles.sum, 'sum')
        self.set_pen(self.styles.residual, 'residual')
        self.set_pen(self.styles.sigma3, 'sigma3')

    def create_deconvoluted_dataset_new(self) -> pd.DataFrame:
        """
        Создает датасет на основе данных разделенных линий.
        Колонки - амплитуда и x0 (опционально)
        Строки - значения для каждого спектра

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        filenames = mw.ui.fit_params_table.model().filenames
        x_axis = self.graph_drawing.array_of_current_filename_in_deconvolution()[:, 0]
        line_legends = mw.ui.deconv_lines_table.model().column_data(0).sort_values()
        line_indexes = line_legends.index.values
        line_legends = line_legends.values
        line_legends_target_param, line_legends_x0 = [], []
        use_area = mw.ui.use_area_check_box.isChecked()
        target_char = 's' if use_area else 'a'
        include_x0 = mw.ui.include_x0_checkBox.isChecked()
        for i in line_legends:
            line_legends_target_param.append(i + f'_{target_char}')
            if include_x0:
                line_legends_x0.append(i + '_x0')
        line_legends = np.concatenate((line_legends_target_param, line_legends_x0)) \
            if include_x0 else line_legends_target_param
        params = mw.ui.fit_params_table.model().column_data(1)
        idx_line_type = mw.ui.deconv_lines_table.model().column_data(1)
        df = DataFrame(columns=line_legends)
        class_ids = []
        if mw.predict_logic.is_production_project:
            for _ in filenames:
                class_ids.append(0)
        filename_group = mw.ui.input_table.model().column_data(2)
        for filename in filenames:
            if not mw.predict_logic.is_production_project:
                try:
                    group_id = filename_group.loc[filename]
                    class_ids.append(group_id)
                except KeyError:
                    class_ids.append(1)
            if use_area:
                values_target_param = []
                for i in line_indexes:
                    y, x, _ = self.graph_drawing.curve_y_x_idx(idx_line_type[i],
                                                               params.loc[(filename, i)], x_axis, i)
                    values_target_param.append(np.trapz(y, x))
            else:
                values_target_param = [params.loc[(filename, i, 'a')] for i in line_indexes]
            values = np.concatenate((values_target_param, [params.loc[(filename, i, 'x0')]
                                                           for i in line_indexes])) \
                if include_x0 else values_target_param
            df2 = DataFrame(np.array(values).reshape(1, -1), columns=line_legends)
            df = pd.concat([df, df2], ignore_index=True)
        df2 = DataFrame({'Class': class_ids, 'Filename': filenames})
        df = pd.concat([df2, df], axis=1)
        if mw.predict_logic.is_production_project \
                and mw.ui.deconvoluted_dataset_table_view.model().rowCount() != 0:
            df = pd.concat([mw.ui.deconvoluted_dataset_table_view.model().dataframe(), df])
        df.reset_index(drop=True)
        self.update_ignore_features_table()
        return df

    def update_ignore_features_table(self):
        mw = get_parent(self.parent, "MainWindow")
        features = mw.ui.deconvoluted_dataset_table_view.model().features
        df = DataFrame({'Feature': features}, columns=['Feature', 'Score', 'P value'])
        mw.ui.ignore_dataset_table_view.model().set_dataframe(df)
        mw.ui.ignore_dataset_table_view.model().set_checked({})

    def add_line_params_from_template(self, filename: str | None = None) -> None:
        """
        When selected item in list.
        update parameters for lines of current spectrum filename
        if no parameters for filename - copy from template and update limits of amplitude parameter
        """
        mw = get_parent(self.parent, "MainWindow")
        model = mw.ui.fit_params_table.model()
        model.delete_rows_by_filenames(filename)
        if filename is None:
            filename = self.data.current_spectrum_name
        df_a = mw.ui.fit_params_table.model().dataframe()
        tuples = []
        for i in df_a.index:
            if i[0] == '':
                tuples.append(i)
        df_a = df_a.loc['']
        tuples_b = []
        for _, line_idx, param_name in tuples:
            tuples_b.append((filename, line_idx, param_name))
        df_b = df_a.copy(deep=True)
        mi = MultiIndex.from_tuples(tuples_b, names=('filename', 'line_index', 'param_name'))
        df_b.index = mi
        model.concat_df(df_b)

    def set_parameters_after_fit_for_spectrum(self, fit_result: ModelResult, filename: str) \
            -> None:
        """
        Set fitted parameters value after fitting

        Parameters
        ---------
        fit_result : lmfit.model.ModelResult

        filename : str
            filename of spectrum model was fitted to
        """
        mw = get_parent(self.parent, "MainWindow")
        table_model = mw.ui.fit_params_table.model()
        for key, param in fit_result.params.items():
            idx, param_name = curve_idx_from_par_name(key)
            table_model.set_parameter_value(filename, idx, param_name, 'Value', param.value, False)
            table_model.set_parameter_value(filename, idx, param_name, 'Max value', param.max,
                                            False)
            table_model.set_parameter_value(filename, idx, param_name, 'Min value', param.min,
                                            False)

    # region UI
    def update_template_combo_box(self) -> None:
        """
        template_combo_box always has 0-element 'Average'
        next elements are groups names
        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.template_combo_box.clear()
        mw.ui.template_combo_box.addItem('Average')
        data = self.parent.preprocessing.stages.av_data.data
        if not data or self.parent.group_table.table_widget.model().rowCount() == 0:
            return
        for i in data:
            group_name = self.parent.group_table.table_widget.model().row_data(i - 1)['Group name']
            mw.ui.template_combo_box.addItem(str(i) + '. ' + group_name)

    def _interval_cb_state_changed(self, a0: int) -> None:
        """a0 = 0 is False, a0 = 2 if True"""
        self.plotting.linear_region.setVisible(a0 == 2)
        if a0 == 2:
            self.graph_drawing.cut_data_interval()
        else:
            n_array = self.graph_drawing.array_of_current_filename_in_deconvolution()
            if n_array is None:
                return
            self.curves.data.setData(x=n_array[:, 0], y=n_array[:, 1])
            self.graph_drawing.assign_style('data')
        self.graph_drawing.redraw_curves_for_filename()
        self.graph_drawing.draw_sum_curve()
        self.graph_drawing.draw_residual_curve()

    def _init_linear_region(self):
        cfg = get_config('plots')['preproc']
        mw = get_parent(self.parent, "MainWindow")
        color_for_lr = QColor(environ["secondaryDarkColor"])
        color_for_lr.setAlpha(cfg['linear_region_alpha'])
        color_for_lr_hover = QColor(environ["secondaryDarkColor"])
        color_for_lr_hover.setAlpha(cfg['linear_region_hover_alpha'])
        start, end, start_min, end_max = self._get_range_info()
        self.plotting.linear_region = LinearRegionItem([start, end], bounds=[start_min, end_max],
                                                       swapMode="push")
        self.plotting.linear_region.setBrush(color_for_lr)
        self.plotting.linear_region.setHoverBrush(color_for_lr_hover)
        self.plotting.linear_region.sigRegionChangeFinished.connect(self._lr_deconv_region_changed)
        self.plotting.linear_region.setMovable(not mw.ui.lr_movableBtn.isChecked())
        self.plotting.linear_region.setVisible(False)

    def _lr_deconv_region_changed(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        current_region = self.plotting.linear_region.getRegion()
        mw.ui.interval_start_dsb.setValue(current_region[0])
        mw.ui.interval_end_dsb.setValue(current_region[1])

    def _get_range_info(self) -> tuple:
        mw = get_parent(self.parent, "MainWindow")
        return (mw.ui.interval_start_dsb.value(), mw.ui.interval_end_dsb.value(),
                mw.ui.interval_start_dsb.minimum(), mw.ui.interval_end_dsb.maximum())

    def _interval_start_changed(self, new_value: float) -> None:
        """
        executing when value of self.ui.interval_start_dsb was changed by user or by code
        self.old_start_interval_value contains previous value

        Parameters
        ----------
        new_value : float
            New value of self.ui.interval_start_dsb
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        context.set_modified()
        corrected_value = None
        # correct value - take nearest from x_axis
        av_data = context.preprocessing.stages.av_data.data
        if av_data:
            x_axis = next(iter(av_data.values()))[:, 0]
            corrected_value = find_nearest(x_axis, new_value)
        if corrected_value is not None and round(corrected_value, 5) != new_value:
            mw.ui.interval_start_dsb.setValue(corrected_value)
            return
        if new_value >= mw.ui.interval_end_dsb.value():
            mw.ui.interval_start_dsb.setValue(mw.ui.interval_start_dsb.minimum())
            return
        self.plotting.linear_region.setRegion(
            (mw.ui.interval_start_dsb.value(), mw.ui.interval_end_dsb.value()))
        # if interval checked - change cut interval
        if mw.ui.interval_checkBox.isChecked():
            self.graph_drawing.cut_data_interval()
            self.graph_drawing.redraw_curves_for_filename()
            self.graph_drawing.draw_sum_curve()
            self.graph_drawing.draw_residual_curve()

    def _interval_end_changed(self, new_value: float) -> None:
        """
        executing when value of self.ui.interval_end_dsb was changed by user or by code
        self.old_end_interval_value contains previous value

        Parameters
        ----------
        new_value : float
            New value of self.ui.interval_end_dsb
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        context.set_modified()
        corrected_value = None
        # correct value - take nearest from x_axis
        av_data = context.preprocessing.stages.av_data.data
        if av_data:
            x_axis = next(iter(av_data.values()))[:, 0]
            corrected_value = find_nearest(x_axis, new_value)
        if corrected_value is not None and round(corrected_value, 5) != new_value:
            mw.ui.interval_end_dsb.setValue(corrected_value)
            return
        if new_value <= mw.ui.interval_start_dsb.value():
            mw.ui.interval_end_dsb.setValue(mw.ui.interval_end_dsb.minimum())
            return
        self.plotting.linear_region.setRegion(
            (mw.ui.interval_start_dsb.value(), mw.ui.interval_end_dsb.value()))
        # if interval checked - change cut interval
        if mw.ui.interval_checkBox.isChecked():
            self.graph_drawing.cut_data_interval()
            self.graph_drawing.redraw_curves_for_filename()
            self.graph_drawing.draw_sum_curve()
            self.graph_drawing.draw_residual_curve()

    def _initial_add_line_button(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        add_lines_menu = QMenu()
        for line_type in get_config('fitting')['peak_shape_names']:
            action = add_lines_menu.addAction(line_type)
            action.triggered.connect(
                lambda checked=None, line=line_type: self._add_deconv_line(line_type=line)
            )
        mw.ui.add_line_button.setMenu(add_lines_menu)
        mw.ui.add_line_button.menu()

    def _initial_guess_button(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        guess_menu = QMenu()
        for line_type in get_config('fitting')['peak_shape_names']:
            action = guess_menu.addAction(line_type)
            action.triggered.connect(
                lambda checked=None, line=line_type: self.b.guess(line_type=line)
            )
        mw.ui.guess_button.setMenu(guess_menu)
        mw.ui.guess_button.menu()

    def _data_cb_state_changed(self, a0: int) -> None:
        """
        Set data_curve visible if a0 == 2

        Parameters
        ----------
        a0: int
            2 is checked, 0 is unchecked

        Returns
        -------
        out : None

        """
        if self.curves.data is None:
            return
        self.curves.data.setVisible(a0 == 2)

    def _sum_cb_state_changed(self, a0: int) -> None:
        """
        Set sum_curve visible if a0 == 2

        Parameters
        ----------
        a0: int
            2 is checked, 0 is unchecked

        Returns
        -------
        out : None

        """
        if self.curves.sum is None:
            return
        self.curves.sum.setVisible(a0 == 2)

    def _sigma3_cb_state_changed(self, a0: int) -> None:
        """
        Set sigma3 fill visible if a0 == 2

        Parameters
        ----------
        a0: int
            2 is checked, 0 is unchecked

        Returns
        -------
        out : None
        """
        if self.curves.sigma3_fill is None:
            return
        self.curves.sigma3_fill.setVisible(a0 == 2)

    def _residual_cb_state_changed(self, a0: int) -> None:
        """
        Set residual_curve visible if a0 == 2

        Parameters
        ----------
        a0: int
            2 is checked, 0 is unchecked

        Returns
        -------
        out : None

        """
        if self.curves.residual is None:
            return
        self.curves.residual.setVisible(a0 == 2)

    def _style_pb_clicked(self, style_type: str) -> None:
        match style_type:
            case 'data':
                d = self.curves.data
                s = self.styles.data
                idx = 999
            case 'sum':
                d = self.curves.sum
                s = self.styles.sum
                idx = 998
            case 'residual':
                d = self.curves.residual
                s = self.styles.residual
                idx = 997
            case 'sigma3':
                d = self.data.sigma3
                s = self.styles.sigma3
                idx = 996
            case _:
                return
        if d is None:
            return
        for obj in get_objects():
            if isinstance(obj, CurvePropertiesWindow) and obj.idx() == idx and obj.isVisible():
                return
        data_curve_prop_window = CurvePropertiesWindow(get_parent(self.parent, "MainWindow"),
                                                       s, idx, False)
        data_curve_prop_window.sigStyleChanged.connect(lambda style, old_style, _, st=style_type:
                                                       self._update_curve_style(style, old_style,
                                                                                st))
        data_curve_prop_window.show()

    def _update_curve_style(self, style: dict, old_style: dict, style_type: str) -> None:
        context = get_parent(self.parent, "Context")
        command = CommandUpdateDataCurveStyle((style, old_style), context,
                                              text=f"Update style for {style_type} curve",
                                              **{'stage': self, 'curve_type': style_type})
        context.undo_stack.push(command)

    def _style_button_style_sheet(self, hex_color: str, curve_type: str) -> None:
        mw = get_parent(self.parent, "MainWindow")
        css_text = f"""*{{background-color: {hex_color};}}"""
        match curve_type:
            case 'data':
                mw.ui.data_pushButton.setStyleSheet(css_text)
            case 'sum':
                mw.ui.sum_pushButton.setStyleSheet(css_text)
            case 'residual':
                mw.ui.residual_pushButton.setStyleSheet(css_text)
            case 'sigma3':
                mw.ui.sigma3_pushButton.setStyleSheet(css_text)

    def set_pen(self, style, curve_type) -> None:
        color = style['color']
        color.setAlphaF(1.0)
        if curve_type == 'data':
            self.styles.data = style
        elif curve_type == 'sum':
            self.styles.sum = style
        elif curve_type == 'residual':
            self.styles.residual = style
        elif curve_type == 'sigma3':
            self.styles.sigma3 = style
        self.graph_drawing.assign_style(curve_type)
        self._style_button_style_sheet(color.name(), curve_type)

    # endregion

    # region Add line
    def _add_deconv_line(self, line_type: str):
        mw = get_parent(self.parent, "MainWindow")
        stage = mw.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if not self.data.is_template:
            msg = MessageBox("Add line failed.", "Switch to Template mode to add new line",
                             mw, {"Ok"})
            msg.exec()
            return
        if not data:
            msg = MessageBox("Add line failed.", "No baseline corrected spectrum", mw, {"Ok"})
            msg.exec()
            return
        self._do_add_deconv_line(line_type)

    def _do_add_deconv_line(self, line_type: str) -> None:
        mw = get_parent(self.parent, "MainWindow")
        idx = mw.ui.deconv_lines_table.model().free_index()
        context = get_parent(self.parent, "Context")
        command = CommandAddDeconvLine((idx, line_type), context, text=f"Add {line_type}")
        context.undo_stack.push(command)

    # endregion

    def switch_template(self, current_text: str | bool = 'Average') -> None:
        """
        Update data for selected averaged spectrum (template).
        """
        print(f'switch_template {current_text}')
        if isinstance(current_text, bool):
            current_text = 'Average'
        self.graph_drawing.update_single_deconvolution_plot(current_text, True, True)
        self.graph_drawing.redraw_curves_for_filename()
        self.show_all_roi()
        self.set_rows_visibility()
        self.show_current_report_result()
        self.graph_drawing.draw_sum_curve()
        self.graph_drawing.draw_residual_curve()
        self.graph_drawing.update_sigma3_curves('')

    def sum_array(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return x, y arrays of sum spectra of all visible fit curves

        Returns
        -------
        out : tuple[np.ndarray, np.ndarray]
            x_axis, y_axis of sum curve
        """
        mw = get_parent(self.parent, "MainWindow")
        x_axis = self.get_x_axis()
        if mw.ui.interval_checkBox.isChecked():
            x_axis, _, _ = cut_axis(x_axis, mw.ui.interval_start_dsb.value(),
                                    mw.ui.interval_end_dsb.value())
        data_items = mw.ui.deconv_plot_widget.getPlotItem().listDataItems()
        y_axis = np.zeros(x_axis.shape[0])

        for i in data_items:
            if not (isinstance(i, PlotCurveItem) and i.isVisible()):
                continue
            x, y = i.getData()
            idx = nearest_idx(x_axis, x[0])
            y_z = np.zeros(x_axis.shape[0])
            if x_axis.shape[0] < y.shape[0]:
                idx_right = x_axis.shape[0] - idx - 1
                y_z[idx: idx + idx_right] += y[:idx_right]
            else:
                y_z[idx: idx + y.shape[0]] += y
            y_axis += y_z
        return x_axis, y_axis

    def get_x_axis(self) -> np.ndarray:
        mw = get_parent(self.parent, "MainWindow")
        stage = mw.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        return next(iter(data.values()))[:, 0]

    def current_line_parameters(self, index: int, filename: str | None = None) -> dict | None:
        mw = get_parent(self.parent, "MainWindow")
        if filename is None:
            filename = "" if self.data.is_template or self.data.current_spectrum_name == '' \
                else self.data.current_spectrum_name
        if filename not in mw.ui.fit_params_table.model().filenames and not filename == "":
            return None
        df_params = mw.ui.fit_params_table.model().get_df_by_multiindex((filename, index))
        line_type = mw.ui.deconv_lines_table.model().cell_data_by_idx_col_name(index, 'Type')
        if df_params.empty:
            return None
        return packed_current_line_parameters(df_params, line_type, peak_shapes_params())

    def show_all_roi(self):
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        for i in plot_item.listDataItems():
            if isinstance(i, PlotCurveItem):
                i.parentItem().show()

    def _hide_all_roi(self):
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        for i in plot_item.listDataItems():
            if isinstance(i, PlotCurveItem):
                i.parentItem().hide()

    def set_rows_visibility(self) -> None:
        """
        Show only rows of selected curve in fit_params_table. Other rows hiding.
        If not selected - show first curve's params
        """
        mw = get_parent(self.parent, "MainWindow")
        row_count = mw.ui.fit_params_table.model().rowCount()
        if row_count == 0:
            return
        filename = '' if self.data.is_template else self.data.current_spectrum_name
        row_line = mw.ui.deconv_lines_table.selectionModel().currentIndex().row()
        row = row_line if row_line != -1 else 0
        idx = mw.ui.deconv_lines_table.model().row_data(row).name
        row_id_to_show = mw.ui.fit_params_table.model().row_number_for_filtering(
            (filename, idx))
        if row_id_to_show is None:
            return
        for i in range(row_count):
            mw.ui.fit_params_table.setRowHidden(i, True)
        for i in row_id_to_show:
            mw.ui.fit_params_table.setRowHidden(i, False)

    def show_current_report_result(self) -> None:
        """
        Show report result of currently showing spectrum name
        """
        mw = get_parent(self.parent, "MainWindow")
        filename = '' if self.data.is_template else self.data.current_spectrum_name
        report = self.data.report_result[filename] if filename in self.data.report_result else ''
        mw.ui.report_text_edit.setText(report)

    def remove_all_lines_from_plot(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.deconv_plot_widget.getPlotItem()
        data_items = plot_item.listDataItems()
        if len(data_items) == 0:
            return
        items_matches = (x for x in data_items if isinstance(x.name(), int))
        for i in items_matches:
            plot_item.removeItem(i.parentItem())
            plot_item.removeItem(i)
        plot_item.addItem(self.plotting.linear_region)
        if self.curves.sigma3_fill is not None:
            plot_item.addItem(self.curves.sigma3_fill)
        plot_item.getViewBox().updateAutoRange()

    def dec_table_double_clicked(self):
        """
        When selected item in list.
        Change current spectrum in deconv_plot_widget
        """
        mw = get_parent(self.parent, "MainWindow")
        current_index = mw.ui.dec_table.selectionModel().currentIndex()
        current_spectrum_name = mw.ui.dec_table.model().cell_data(current_index.row())
        if current_spectrum_name not in mw.ui.fit_params_table.model().filenames:
            self._hide_all_roi()
        else:
            self.show_all_roi()
        self.data.current_spectrum_name = current_spectrum_name
        self.graph_drawing.update_single_deconvolution_plot(current_spectrum_name)
        self.graph_drawing.redraw_curves_for_filename()
        self.set_rows_visibility()
        self.graph_drawing.draw_sum_curve()
        self.graph_drawing.draw_residual_curve()
        self.show_current_report_result()
        self.graph_drawing.update_sigma3_curves(current_spectrum_name)

    def initial_peak_parameters(self, line_type: str) -> dict:
        """
        The function returns dict with initial parameters and x_axis.
            'x_axis': np.ndarray, 'a': float, 'x0': float, 'dx': float, + additional parameters

        Parameters
        ----------
        line_type: str
            input array to slice

        line_type: str
            {'Gaussian', 'Split Gaussian', ... etc.}. Line type chosen by user in Guess button.
             Look peak_shapes_params() in default_values.py

        Returns
        -------
        out : dict
            'x_axis': np.ndarray, 'a': float, 'x0': float, 'dx': float, + additional parameters
        """
        mw = get_parent(self.parent, "MainWindow")
        x_axis, a, x0, dx, arr = np.array(range(920, 1080)), 100.0, 1000.0, 10.0, None
        stage = mw.drag_widget.get_latest_active_stage()
        assert stage is not None, 'Cant find latest active stage.'
        data = stage.data
        if mw.ui.template_combo_box.currentText() == 'Average':
            arr = self.data.averaged_spectrum
            dx = np.max(mw.ui.input_table.model().column_data(6)) * np.pi / 2.
        elif self.data.current_spectrum_name:
            arr = data[self.data.current_spectrum_name]
            row_data = mw.ui.input_table.model().row_data_by_index(self.data.current_spectrum_name)
            dx = row_data['FWHM, cm\N{superscript minus}\N{superscript one}'] * np.pi / 2.
        elif not self.data.current_spectrum_name \
                and mw.ui.template_combo_box.currentText() != 'Average':
            array_id = int(mw.ui.template_combo_box.currentText().split('.')[0])
            arr = self.parent.preprocessing.stages.av_data.data[array_id]
            dx = np.max(mw.ui.input_table.model().column_data(6)) * np.pi / 2.

        if arr is not None:
            x_axis, a = arr[:, 0], np.max(arr[:, 1]) / 2
            x0 = np.mean(x_axis)

        result = {'x_axis': x_axis, 'a': np.round(a, 5),
                  'x0': np.round(x0, 5), 'dx': np.round(dx, 5)}
        if 'add_params' not in peak_shapes_params()[line_type]:
            return result
        # add additional parameters into result dict
        add_params = peak_shapes_params()[line_type]['add_params']
        for param_name in add_params:
            match param_name:
                case 'dx_left':
                    result[param_name] = np.round(dx - dx * 0.1, 5)
                case 'gamma':
                    result[param_name] = 0.5
                case 'skew':
                    result[param_name] = 0.0
                case 'l_ratio':
                    result[param_name] = 0.6
                case 'expon' | 'beta':
                    result[param_name] = 1.0
                case 'alpha' | 'q':
                    result[param_name] = 0.1
        return result

    def _set_deconvoluted_dataset(self) -> None:
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        if (
                mw.ui.input_table.model().rowCount() == 0
                or mw.ui.fit_params_table.model().batch_unfitted()
        ):
            return
        df = self.create_deconvoluted_dataset_new()
        mw.ui.deconvoluted_dataset_table_view.model().set_dataframe(df)
        context.datasets.init_current_filename_combobox()
        self.update_ignore_features_table()
