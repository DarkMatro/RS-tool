from qtpy.QtWidgets import QHeaderView, QAbstractItemView, QLineEdit, QMenu
from qtpy.QtCore import QPointF
from pandas import MultiIndex, DataFrame

from qfluentwidgets import MessageBox
from src import get_parent
from src.data.default_values import peak_shapes_params
from src.pandas_tables import PandasModelFitParamsTable, DoubleSpinBoxDelegate
from src.stages.fitting.functions.plotting import deconvolution_data_items_by_idx, set_roi_size


class TableParams:
    def __init__(self, parent):
        self.parent = parent
        self.reset()
        self.set_ui()

    def reset(self):
        mw = get_parent(self.parent, "MainWindow")
        multi_index = MultiIndex.from_tuples(
            [("", 0, "a")], names=("filename", "line_index", "param_name")
        )
        df = DataFrame(
            columns=["Parameter", "Value", "Min value", "Max value"], index=multi_index
        )
        model = PandasModelFitParamsTable(self, df)
        mw.ui.fit_params_table.setModel(model)
        mw.ui.fit_params_table.model().clear_dataframe()

    def set_ui(self):
        mw = get_parent(self.parent, "MainWindow")
        dsb_delegate = DoubleSpinBoxDelegate(mw)
        for i in range(1, 4):
            mw.ui.fit_params_table.setItemDelegateForColumn(i, dsb_delegate)
        dsb_delegate.sigLineParamChanged.connect(self.curve_parameter_changed)
        mw.ui.fit_params_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        mw.ui.fit_params_table.horizontalHeader().resizeSection(0, 70)
        mw.ui.fit_params_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Interactive
        )
        mw.ui.fit_params_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        mw.ui.fit_params_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        mw.ui.fit_params_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        mw.ui.fit_params_table.contextMenuEvent = self._deconv_params_table_context_menu_event
        mw.ui.fit_params_table.verticalHeader().setVisible(False)

    def curve_parameter_changed(self, value: float, line_index: int, param_name: str) -> None:
        mw = get_parent(self.parent, "MainWindow")
        data_items = mw.ui.deconv_plot_widget.getPlotItem().listDataItems()
        items_matches = deconvolution_data_items_by_idx(line_index, data_items)
        if items_matches is None:
            return
        curve, roi = items_matches
        line_type = mw.ui.deconv_lines_table.model().cell_data_by_idx_col_name(line_index, "Type")
        if param_name == "x0":
            new_pos = QPointF()
            new_pos.setX(value)
            new_pos.setY(roi.pos().y())
            roi.setPos(new_pos)
        elif param_name == "a":
            set_roi_size(roi.size().x(), value, roi, [0, 0])
        elif param_name == "dx":
            set_roi_size(value, roi.size().y(), roi)
        params = {
            "a": value if param_name == "a" else roi.size().y(),
            "x0": roi.pos().x(), "dx": roi.size().x(),
        }
        filename = "" if self.parent.data.is_template else self.parent.data.current_spectrum_name
        model = mw.ui.fit_params_table.model()
        if "add_params" not in peak_shapes_params()[line_type]:
            self.parent.graph_drawing.redraw_curve(params, curve, line_type)
            return
        add_params = peak_shapes_params()[line_type]["add_params"]
        for s in add_params:
            if param_name == s:
                param = value
            else:
                param = model.get_parameter_value(filename, line_index, s, "Value")
            params[s] = param
        self.parent.graph_drawing.redraw_curve(params, curve, line_type)

    def _deconv_params_table_context_menu_event(self, a0) -> None:
        mw = get_parent(self.parent, "MainWindow")
        if self.parent.data.is_template:
            return
        line = QLineEdit(mw)
        menu = QMenu(line)
        menu.addAction("Copy line parameters from template",
                       self.copy_line_parameters_from_template)
        menu.move(a0.globalPos())
        menu.show()

    def copy_line_parameters_from_template(self, idx: int | None = None,
                                           filename: str | None = None,
                                           redraw: bool = True) -> None:
        mw = get_parent(self.parent, "MainWindow")
        filename = self.parent.data.current_spectrum_name if filename is None else filename
        model = self.parent.ui.fit_params_table.model()
        # find current index of selected line
        if idx is None:
            row = mw.ui.deconv_lines_table.selectionModel().currentIndex().row()
            if row == -1:
                MessageBox("Line isn't selected", 'Select line', mw, {'Ok'}).exec()
                return
            idx = mw.ui.deconv_lines_table.model().row_data(row).name
            # delete line params by index
            model.delete_rows_multiindex((filename, idx))
        # add line params by from template
        mi = '', idx
        df = model.get_df_by_multiindex(mi)
        for i in range(len(df)):
            row_data = df.iloc[i]
            model.append_row(idx, row_data.Parameter, row_data.Value, row_data['Min value'],
                             row_data['Max value'], filename)
        if redraw:
            self.parent.graph_drawing.redraw_curve_by_index(idx)
