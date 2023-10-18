import numpy as np
from pandas import DataFrame, MultiIndex, concat, ExcelWriter, Series
from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal, Slot as pyqtSlot, QPoint
from qtpy.QtGui import QColor, QCursor
from qtpy.QtWidgets import QDoubleSpinBox, QStyledItemDelegate, QListWidget, QListWidgetItem
from qfluentwidgets import TableItemDelegate
from modules.default_values import peak_shape_params_limits
from modules.undo_redo import CommandUpdateTableCell, CommandDeconvLineParameterChanged, CommandFitIntervalChanged


class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """ Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if self._dataframe is not None and parent == QModelIndex() and not self._dataframe.empty:
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if self._dataframe is None:
            return 0
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def clear_dataframe(self) -> None:
        self._dataframe = self._dataframe[0:0]
        self.modelReset.emit()

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            if isinstance(value, QColor):
                return None
            else:
                return str(value)
        elif role == Qt.ItemDataRole.EditRole:
            return value
        elif role == Qt.ItemDataRole.DecorationRole and isinstance(value, QColor):
            return QColor(value)
        elif role == Qt.ItemDataRole.ForegroundRole and index.column() + 1 < self.columnCount():
            value_color = self._dataframe.iloc[index.row(), index.column() + 1]
            if isinstance(value_color, QColor):
                return QColor(value_color)
        return None

    def cell_data(self, row: int | str, column: int = 0):
        if isinstance(row, int):
            result = self._dataframe.iloc[row, column]
        else:
            result = self._dataframe.at[row, column]
        return result

    def cell_data_by_idx_col_name(self, index: int, column_name: str) -> dict:
        return self._dataframe.loc[index, column_name]

    def set_cell_data_by_idx_col_name(self, index: int, column_name: str, value: str) -> dict:
        self._dataframe.loc[index, column_name] = value
        self.modelReset.emit()

    def cell_data_by_index(self, index: QModelIndex):
        return self._dataframe.iloc[index.row(), index.column()]

    def set_cell_data_by_index(self, index: QModelIndex, value: str) -> None:
        self._dataframe.iloc[index.row(), index.column()] = value
        self.modelReset.emit()

    def set_cell_data_by_row_column(self, row: int, column: int, value: str) -> None:
        self._dataframe.iloc[row, column] = value
        self.modelReset.emit()

    def row_data(self, row: int):
        return self._dataframe.iloc[row]

    def get_df_by_multiindex(self, mi: MultiIndex) -> DataFrame | None:
        return self._dataframe.loc[mi]

    def query_result(self, q: str) -> DataFrame:
        return self._dataframe.query(q)

    def query_result_with_list(self, q: str, input_list: list) -> DataFrame:
        """

        @param q: 'filename == @input_list'
        @param input_list:
        @return:
        """
        return self._dataframe.query(q)

    def is_query_result_empty(self, q: str) -> bool:
        return self._dataframe.query(q).empty

    def column_data(self, column: int):
        return self._dataframe.iloc[:, column]

    def column_data_by_indexes(self, indexes: list, column: str) -> list:
        return list(self._dataframe[column][indexes])

    def get_column(self, column: str):
        return self._dataframe[column]

    def append_dataframe(self, df2: DataFrame) -> None:
        self._dataframe = concat([self._dataframe, df2])
        self._dataframe = self._dataframe.sort_index()
        self.modelReset.emit()

    def get_group_by_name(self, name: str) -> int:
        row = self._dataframe.loc[self._dataframe.index == name]
        if not row.empty:
            return row.values[0][2]
        else:
            return 0

    def change_cell_data(self, row: int | str, column: int | str, value: QColor | float | dict) -> None:
        if isinstance(row, int):
            self._dataframe.iat[row, column] = value
        else:
            self._dataframe.at[row, column] = value
        self.modelReset.emit()

    def set_dataframe(self, df: DataFrame) -> None:
        self._dataframe = df.copy()
        self.modelReset.emit()

    def get_csv(self) -> str | None:
        return self._dataframe.to_csv('groups.csv')

    def dataframe(self) -> DataFrame:
        return self._dataframe

    def sort_index(self, ascending: bool = True) -> None:
        self._dataframe = self._dataframe.sort_index(ascending=ascending)
        self.modelReset.emit()

    def setData(self, index, value, role):
        if value != '' and role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = value
            self.dataChanged.emit(index, index)
            return True
        else:
            return False

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole) -> str | None:
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])

        return None

    def row_by_index(self, index: int) -> int:
        for i, idx in enumerate(self._dataframe.index):
            if idx == index:
                return i

    def index_by_row(self, row: int) -> int:
        return self._dataframe.iloc[row].name

    def delete_rows_by_filenames(self, filenames: list[str]) -> None:
        """
        Drop table rows by Filename column

        Parameters
        ---------
        filenames : list[str]
            filename, line_index, param_name
        """
        rows = self._dataframe.loc[self._dataframe['Filename'].isin(filenames)]
        self._dataframe = self._dataframe.drop(rows.index)
        self.modelReset.emit()

    def idx_by_column_value(self, column_name: str, value) -> int:
        return self._dataframe.loc[self._dataframe[column_name] == value].index[0]

    def set_column_data(self, col: str, ser: list):
        self._dataframe[col] = Series(ser)
        self.modelReset.emit()

    def sort_values(self, current_name: str, ascending: bool) -> None:
        self._dataframe = self._dataframe.sort_values(by=current_name, ascending=ascending)
        self.modelReset.emit()


class PandasModelInputTable(PandasModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: DataFrame):
        super().__init__(dataframe)
        self._dataframe = dataframe

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            if isinstance(value, QColor):
                return None
            else:
                return str(value)
        return None

    def append_row_input_table(self, name: str, min_nm: float, max_nm: float, despiked_nm: list[float] | str,
                               rayleigh_line: float | str, fwhm: float, fwhm_cm: float = 0, group: int = 0,
                               snr: float | str = '') \
            -> None:
        df2 = DataFrame({'Min, nm': min_nm,
                         'Max, nm': max_nm,
                         'Group': group,
                         'Despiked, nm': despiked_nm,
                         'Rayleigh line, nm': rayleigh_line,
                         'FWHM, nm': fwhm,
                         'FWHM, cm\N{superscript minus}\N{superscript one}': fwhm_cm,
                         'SNR': snr,
                         }, index=name
                        )
        self._dataframe = concat([self._dataframe, df2])
        self._dataframe = self._dataframe.sort_index()
        self.modelReset.emit()

    def delete_rows_input_table(self, names: list[str]) -> None:
        self._dataframe = self._dataframe[~self._dataframe.index.isin(names)]
        self._dataframe = self._dataframe.sort_index()
        self.modelReset.emit()

    def get_filename_by_row(self, row: int) -> str:
        idx = self._dataframe.index[row]
        return idx

    def to_excel(self, folder: str) -> None:
        with ExcelWriter(folder + '\output.xlsx') as writer:
            self._dataframe.to_excel(writer, sheet_name='Spectrum info')

    def flags(self, index):
        if index.column() == 2:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        else:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def sort_values(self, current_name: str, ascending: bool) -> None:
        self._dataframe = self._dataframe.sort_values(by=current_name, ascending=ascending)
        self.modelReset.emit()

    def names_of_group(self, group_number: int) -> list[str]:
        return self._dataframe.loc[self._dataframe['Group'] == group_number].index

    def row_data_by_index(self, idx: str):
        return self._dataframe.loc[idx]

    def min_fwhm(self) -> float:
        return np.min(self.get_column('FWHM, cm\N{superscript minus}\N{superscript one}').values)

    def save_nm_files_to_csv(self, filepath: str, raw_spectra: dict) -> None:
        """
        Function creates dataframe like
        Group_id         | Filename | x_nm_0        | x_nm_1 .... x_nm_n
        int                 str      y_intensity_0  ....  y_intensity_n

        and saves it into filepath.csv file

        Parameters
        ----------
        filepath: str
            Path to save .csv file
        raw_spectra: dict
            key: str - filename
            value: 2D ndarray - x_axis - nanometers, y_axis - intensity

        Returns
        -------
            None
        """
        x_axis = next(iter(raw_spectra.values()))[:, 0]
        nm_params = []
        for i in x_axis:
            nm_params.append('%s' % np.round(i, 2))
        df = DataFrame(columns=nm_params)
        class_ids = []
        filename_group = self._dataframe.iloc[:, 2]
        for filename, n_array in raw_spectra.items():
            class_ids.append(filename_group.loc[filename])
            df2 = DataFrame(n_array[:, 1].reshape(1, -1), columns=nm_params)
            df = concat([df, df2], ignore_index=True)
        indexes = list(filename_group.index)
        df2 = DataFrame({'Class': class_ids, 'Filename': indexes})
        df = concat([df2, df], axis=1)
        df.to_csv(filepath)


class PandasModelGroupsTable(PandasModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self._dataframe = dataframe
        self.parent = parent

    def get_group_name_by_int(self, group_number: int) -> str:
        return self._dataframe.loc[self._dataframe.index == group_number].iloc[0]['Group name']

    def groups_list(self) -> list:
        return self._dataframe.index

    def append_group(self, group: str, style: dict, index: int) -> None:
        df2 = DataFrame({'Group name': [group],
                         'Style': [style]
                         }, index=[index]
                        )
        self._dataframe = concat([self._dataframe, df2])
        self._dataframe = self._dataframe.sort_index()
        self._dataframe = self._dataframe.reset_index(drop=True)
        self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self.modelReset.emit()

    def remove_group(self, row: int) -> None:
        self._dataframe = self._dataframe.drop(row)
        # self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self._dataframe = self._dataframe.sort_index()
        self.modelReset.emit()

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            if isinstance(value, dict):
                enum_id = value['style']
                if enum_id == 1:
                    str_style = 'SolidLine'
                elif enum_id == 2:
                    str_style = 'DashLine'
                elif enum_id == 3:
                    str_style = 'DotLine'
                elif enum_id == 4:
                    str_style = 'DashDotLine'
                elif enum_id == 5:
                    str_style = 'DashDotDotLine'
                elif enum_id == 6:
                    str_style = 'CustomDashLine'
                else:
                    str_style = 'NoPen'
                # str_style = str(value['style'])
                # str_style = str_style.split('.')[-1]
                return str_style + ', %s pt' % value['width']
            else:
                return str(value)
        elif role == Qt.ItemDataRole.EditRole:
            if index.column() == 1:
                return 'changing...'
            else:
                return value
        elif role == Qt.ItemDataRole.DecorationRole and isinstance(value, dict):
            color = value['color']
            color.setAlphaF(1.0)
            return color
        elif role == Qt.ItemDataRole.ForegroundRole and index.column() + 1 < self.columnCount():
            value_color = self._dataframe.iloc[index.row(), index.column() + 1]
            if isinstance(value_color, QColor):
                return QColor(value_color)
        return None

    def setData(self, index, value, role):
        if role == Qt.EditRole and not index.column() == 1:
            old_value = self._dataframe.iloc[index.row(), index.column()]
            command = CommandUpdateTableCell(self, self.parent, index, value, old_value,
                                             'Change group name')
            self.parent.undoStack.push(command)
            return True
        else:
            return False

    @property
    def groups_styles(self) -> list:
        return [x for x in self._dataframe.iloc[:, 1]]

    @property
    def groups_colors(self) -> list:
        colors = []
        for style in self.groups_styles:
            colors.append(style['color'].name())
        return colors

    @property
    def groups_width(self) -> list:
        width = []
        for style in self.groups_styles:
            width.append(style['width'])
        return width

    @property
    def groups_dashes(self) -> list:
        dashes = []
        for style in self.groups_styles:
            match style['style']:
                case Qt.PenStyle.SolidLine:
                    dashes.append('')
                case Qt.PenStyle.DotLine:
                    dashes.append((1, 1))
                case Qt.PenStyle.DashLine:
                    dashes.append((4, 4))
                case Qt.PenStyle.DashDotLine:
                    dashes.append((4, 1.25, 1.5, 1.25))
                case Qt.PenStyle.DashDotDotLine:
                    dashes.append((4, 1.25, 1.5, 1.25, 1.5, 1.25))
                case _:
                    dashes.append('')
        return dashes


class PandasModelDeconvTable(PandasModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: DataFrame):
        super().__init__(dataframe)
        self._dataframe = dataframe

    def append_row_deconv_table(self, filename: str) -> None:
        df2 = DataFrame({'Filename': filename})
        self._dataframe = concat([self._dataframe, df2])
        self._dataframe = self._dataframe.sort_values(by=['Filename'])
        self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self.modelReset.emit()

    def delete_rows_deconv_table(self, names: list[str]) -> None:
        self._dataframe = self._dataframe[~self._dataframe.Filename.isin(names)]
        self._dataframe = self._dataframe.sort_values(by=['Filename'])
        self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self.modelReset.emit()

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            if isinstance(value, QColor):
                return None
            else:
                return str(value)
        return None


class PandasModelFitIntervals(PandasModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: DataFrame):
        super().__init__(dataframe)
        self._dataframe = dataframe

    def append_row(self) -> None:
        df2 = DataFrame({'Border': [0.]})
        self._dataframe = concat([self._dataframe, df2])
        self.sort_by_border()

    def delete_row(self, interval_number: int) -> None:
        self._dataframe = self._dataframe.drop(interval_number)
        self.sort_by_border()

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        elif role == Qt.ItemDataRole.EditRole:
            return float(value)
        return None

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def sort_by_border(self) -> None:
        self._dataframe = self._dataframe.sort_values(by=['Border'])
        self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self.modelReset.emit()

    def add_auto_found_borders(self, borders: np.ndarray) -> None:
        self._dataframe = DataFrame({'Border': borders})
        self.modelReset.emit()


class PandasModelDeconvLinesTable(PandasModel):
    """A model to interface a Qt view with pandas dataframe """

    sigCheckedChanged = Signal(int, bool)

    def __init__(self, parent, dataframe: DataFrame, checked: list[bool]):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe
        self._checked = checked

    def append_row(self, legend: str = 'Curve 1', line_type: str = 'Gaussian', style=None, idx: int = -1) \
            -> int:
        if style is None:
            style = {}
        if idx == -1:
            idx = self.free_index()
        df2 = DataFrame({'Legend': [legend],
                         'Type': [line_type],
                         'Style': [style]}, index=[idx])
        self._dataframe = concat([self._dataframe, df2])
        self._checked.append(True)
        self.modelReset.emit()
        return idx

    @property
    def indexes(self):
        return self._dataframe.index

    def free_index(self) -> int:
        idx = len(self._dataframe)
        if idx in self._dataframe.index:
            while idx in self._dataframe.index:
                idx += 1
        return idx

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            if isinstance(value, dict):
                enum_id = value['style']
                if enum_id == 1:
                    str_style = 'SolidLine'
                elif enum_id == 2:
                    str_style = 'DashLine'
                elif enum_id == 3:
                    str_style = 'DotLine'
                elif enum_id == 4:
                    str_style = 'DashDotLine'
                elif enum_id == 5:
                    str_style = 'DashDotDotLine'
                elif enum_id == 6:
                    str_style = 'CustomDashLine'
                else:
                    str_style = 'NoPen'
                # str_style = str(value['style'])
                # str_style = str_style.split('.')[-1]
                return str_style + ', %s pt' % value['width']
            else:
                return str(value)
        elif role == Qt.ItemDataRole.EditRole:
            if index.column() == 2:
                return 'changing...'
            else:
                return value
        elif role == Qt.ItemDataRole.DecorationRole and isinstance(value, dict):
            color = value['color']
            color.setAlphaF(1.0)
            return color
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignVCenter + Qt.AlignmentFlag.AlignLeft

        if role == Qt.CheckStateRole and index.column() == 0:
            checked_row = self._dataframe.index[index.row()]
            checked = self._checked[checked_row]
            return Qt.Checked if checked else Qt.Unchecked
        return None

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsUserCheckable

    def setData(self, index, value, role):
        if role == Qt.EditRole and index.column() == 0:
            old_value = self._dataframe.iloc[index.row(), index.column()]
            command = CommandUpdateTableCell(self, self.parent, index, value, old_value,
                                             'Change line name')
            self.parent.undoStack.push(command)
            return True
        elif role == Qt.CheckStateRole:
            checked = value == 2
            checked_idx = self._dataframe.iloc[index.row()].name
            self._checked[checked_idx] = checked
            self.sigCheckedChanged.emit(checked_idx, checked)
            return True
        else:
            return False

    def sort_values(self, current_name: str, ascending: bool) -> None:
        self._dataframe = self._dataframe.sort_values(by=current_name, ascending=ascending)
        self.modelReset.emit()

    def checked(self) -> list[bool]:
        return self._checked

    def clear_dataframe(self) -> None:
        self._dataframe = self._dataframe[0:0]
        self._checked = []
        self.modelReset.emit()

    def set_checked(self, checked: list[bool]) -> None:
        self._checked = checked

    def delete_row(self, idx: int) -> None:
        self._dataframe = self._dataframe.drop(index=idx)
        self.modelReset.emit()

    def get_visible_line_types(self) -> DataFrame:
        """
        get_visible_line_types()

        Return dataframe of lines where checked is True

        Returns
        -------
        out : DataFrame
            Pandas DataFrame

        Examples
        --------
        >>> get_visible_line_types()
            Legend      Type                                              Style
        0  Curve 1     Gaussian                {'color': <PyQt6.QtGui.QColor object at 0x0000...
        """
        idx_list = []
        for i, b in enumerate(self.checked()):
            if b:
                idx_list.append(i)
        return self._dataframe[self._dataframe.index.isin(idx_list)]

    def to_excel(self, folder: str) -> None:
        with ExcelWriter(folder + '\output.xlsx') as writer:
            self._dataframe.to_excel(writer, sheet_name='Fit lines')


class PandasModelFitParamsTable(PandasModel):

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    @property
    def filenames(self) -> list[str]:
        filenames = self._dataframe.index.levels[0].values
        filenames = [i for i in filenames if i]
        return filenames

    def append_row(self, line_index: int, param_name: str, param_value: float, min_v: float = None,
                   max_v: float = None, filename: str = '') -> None:
        # limits in self.parent.deconv_line_params_limits
        min_value = param_value
        max_value = param_value
        par_limits = peak_shape_params_limits()
        if param_name == 'a':
            min_value = 0
            max_value = param_value * 2
        elif param_name == 'x0':
            min_value = param_value - 1
            max_value = param_value + 1
        elif param_name == 'dx':
            min_value = np.round(param_value - param_value / np.pi, 5)
            max_value = np.round(param_value + param_value / np.pi, 5)
        elif param_name in par_limits:
            min_value = par_limits[param_name][0]
            max_value = par_limits[param_name][1]
        min_value = min_value if not min_v else min_v
        max_value = max_value if not max_v else max_v
        tuples = [(filename, line_index, param_name)]
        multi_index = MultiIndex.from_tuples(tuples, names=('filename', 'line_index', 'param_name'))
        df2 = DataFrame({'Parameter': param_name,
                         'Value': param_value,
                         'Min value': min_value,
                         'Max value': max_value}, index=multi_index)
        self._dataframe = concat([self._dataframe, df2])
        self.modelReset.emit()

    def clear_dataframe(self) -> None:
        self._dataframe = self._dataframe[0:0]
        self.modelReset.emit()

    def set_parameter_value(self, filename: str, line_index: int, parameter_name: str, column_name: str,
                            value: float, emit: bool = True) -> None:
        """
        Set value of current cell.

        Parameters
        ---------
        filename : str
            spectrum filename - 0 idx in MultiIndex

        line_index : int
            curve index - 1 idx in MultiIndex

        parameter_name : str
            parameter name - 2 idx in MultiIndex (a, x0, dx, etc.)

        column_name : str
            Value, Min value or Max value

        value : float
            Value to set

        emit : bool
            emit changes or not

        """
        self._dataframe.loc[(filename, line_index, parameter_name), column_name] = np.round(value, 5)
        if emit:
            self.modelReset.emit()

    def model_reset_emit(self) -> None:
        self.modelReset.emit()

    def get_parameter_value(self, filename: str, line_index: int, parameter_name: str, column_name: str) -> float:
        return self._dataframe.loc[(filename, line_index, parameter_name), column_name]

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled
        else:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        elif role == Qt.ItemDataRole.EditRole:
            double_spinbox = QDoubleSpinBox()
            double_spinbox.setValue(float(value))
            double_spinbox.setDecimals(5)
            return float(value)
        return None

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True
        else:
            return False

    def delete_rows(self, idx: int, level: int = 1) -> None:
        """idx means line_index in MultiIndex (filename, line_index, param_name)"""
        self._dataframe = self._dataframe.drop(index=idx, level=level)
        self.modelReset.emit()

    def delete_rows_by_filenames(self, mi: list[str]) -> None:
        """
        Drop table rows by MultiIndex

        Parameters
        ---------
        mi : MultiIndex
            filename, line_index, param_name
        """
        df_index = self._dataframe.index
        existing_mi = []
        for i in mi:
            if i in df_index:
                existing_mi.append(i)
        self._dataframe = self._dataframe.drop(index=existing_mi, level=0)
        self.modelReset.emit()

    def delete_rows_multiindex(self, mi: tuple) -> None:
        """
        Drop table rows by MultiIndex

        Parameters
        ---------
        mi : MultiIndex
            filename, line_index, param_name
        """
        self._dataframe = self._dataframe.drop(index=mi, inplace=True, errors='ignore')
        self.modelReset.emit()

    def concat_df(self, df: DataFrame) -> None:
        """
        self._dataframe + df

        Parameters
        ---------
        df : DataFrame
        """
        self._dataframe = concat([self._dataframe, df])
        self.modelReset.emit()

    def left_only_template_data(self) -> None:
        self._dataframe = self._dataframe.query('filename == ""')
        self.modelReset.emit()

    def to_excel(self, folder: str) -> None:
        with ExcelWriter(folder + '\output.xlsx') as writer:
            self._dataframe.to_excel(writer, sheet_name='Fit params')

    def get_df_by_filename(self, filename: str) -> DataFrame | None:
        filename_indexes = []
        for filename, _, _ in self._dataframe.index:
            if filename not in filename_indexes:
                filename_indexes.append(filename)
        if filename not in filename_indexes:
            return None
        df = self._dataframe.loc[filename]
        if df.empty:
            return None
        return self._dataframe.loc[filename]

    def get_lines_indexes_by_filename(self, filename: str) -> list[int] | None:
        if self._dataframe is None or self._dataframe.empty:
            return None
        df_index = self._dataframe.index
        if filename not in df_index:
            return None
        df = self._dataframe.loc[filename]
        if df.empty:
            return None
        indexes = []
        for idx, _ in df.index:
            if idx not in indexes:
                indexes.append(idx)
        if len(indexes) == 0:
            return None
        return indexes

    def row_number_for_filtering(self, mi: tuple[str, int]):
        df_index = self._dataframe.index
        if mi[0] not in df_index or mi not in df_index:
            return None
        row_count = len(self._dataframe.loc[mi])
        list_idx = list(self._dataframe.index)
        row_num = list_idx.index((mi[0], mi[1], 'a'))
        return range(row_num, row_num + row_count)


class PandasModelSmoothedDataset(PandasModel):

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        else:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        elif role == Qt.ItemDataRole.EditRole:
            double_spinbox = QDoubleSpinBox()
            double_spinbox.setValue(float(value))
            double_spinbox.setDecimals(5)
            return float(value)
        return None

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True
        else:
            return False

    @property
    def filenames(self) -> list[str]:
        filenames = self._dataframe['Filename']
        filenames = [i for i in filenames if i]
        return filenames


class PandasModelBaselinedDataset(PandasModel):

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        else:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        elif role == Qt.ItemDataRole.EditRole:
            double_spinbox = QDoubleSpinBox()
            double_spinbox.setValue(float(value))
            double_spinbox.setDecimals(5)
            return float(value)
        return None

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True
        else:
            return False

    @property
    def filenames(self) -> list[str]:
        filenames = self._dataframe['Filename']
        filenames = [i for i in filenames if i]
        return filenames


class PandasModelDeconvolutedDataset(PandasModel):

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        else:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        elif role == Qt.ItemDataRole.EditRole:
            double_spinbox = QDoubleSpinBox()
            double_spinbox.setValue(float(value))
            double_spinbox.setDecimals(5)
            return float(value)
        return None

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True
        else:
            return False

    @property
    def filenames(self) -> list[str]:
        filenames = self._dataframe['Filename']
        filenames = [i for i in filenames if i]
        return filenames

    @property
    def features(self) -> list[str]:
        return list(self._dataframe.columns.values[2:])

    def delete_rows_by_filenames(self, filenames: list[str]) -> None:
        """
        Drop table rows by Filename column

        Parameters
        ---------
        filenames : list[str]
            filename, line_index, param_name
        """
        rows = self._dataframe.loc[self._dataframe['Filename'].isin(filenames)]
        self._dataframe = self._dataframe.drop(rows.index)
        self.modelReset.emit()
        self._dataframe.reset_index(drop=True, inplace=True)

    def describe_by_group(self, group_id: int) -> DataFrame:
        return self._dataframe[self._dataframe['Class'] == group_id].describe().iloc[:, 1:]


class PandasModelIgnoreDataset(PandasModel):

    def __init__(self, parent, dataframe: DataFrame, checked: dict):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe
        self._checked = checked

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        elif role == Qt.ItemDataRole.EditRole:
            return str(value)
        if role == Qt.CheckStateRole:
            checked_name = self._dataframe.iloc[index.row()].Feature
            return Qt.Checked if checked_name not in self._checked or self._checked[checked_name] else Qt.Unchecked
        return None

    def setData(self, index, value, role):
        if role == Qt.CheckStateRole:
            checked = value == 2
            checked_name = self._dataframe.iloc[index.row()].Feature
            self._checked[checked_name] = checked
            return True
        else:
            return False

    @property
    def checked(self) -> dict:
        return self._checked

    def set_checked(self, checked: dict) -> None:
        self._checked = checked

    def set_all_features_checked(self) -> None:
        """
        self._checked has features which was changed and has structure like {'k1001.00_a': False,  ....}
        This function sets all values to True

        Returns
        -------
        None
        """
        for k in self._checked.keys():
            self._checked[k] = True

    @property
    def ignored_features(self) -> list[str]:
        return [k for k, v in self._checked.items() if not v]

    def features_by_order(self) -> list[str]:
        """
        Returns sorted by Score activated features.
        """
        df_sort = self._dataframe.sort_values(by='Score', ascending=False)
        active_features = [k for k, v in self._checked.items() if v]
        return df_sort['Feature'][df_sort['Feature'].isin(active_features)]


class PandasModelDescribeDataset(PandasModel):

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, index):
        return Qt.ItemIsSelectable

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(np.round(value, 5))
        return None

    def setData(self, index, value, role):
        return False


class PandasModelPredictTable(PandasModel):

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, index):
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        else:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        elif role == Qt.ItemDataRole.ForegroundRole and index.column() != 0:
            class_str = str(value).split(' ')[0]
            return self.parent.get_color_by_group_number(class_str)
        return None

    def setData(self, index, value, role):
        return False


class PandasModelPCA(PandasModel):

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        return None


class DoubleSpinBoxDelegate(QStyledItemDelegate):
    sigLineParamChanged = Signal(float, int, str)

    def __init__(self, rs):
        super().__init__()
        self.RS = rs

    def createEditor(self, parent, option, index):
        param_name = self.RS.ui.fit_params_table.model().row_data(index.row()).name[2]
        min_limit = -1e6
        max_limit = 1e6
        if param_name == 'l_ratio':
            min_limit = 0.0
            max_limit = 1.0
        elif param_name == 'expon' or param_name == 'beta':
            min_limit = 0.1
        elif param_name == 'alpha':
            min_limit = -1.0
            max_limit = 1.0
        elif param_name == 'q':
            min_limit = -0.5
            max_limit = 0.5
        editor = QDoubleSpinBox(parent)
        editor.setDecimals(5)
        editor.setSingleStep(0.1)
        editor.setMinimum(min_limit)
        editor.setMaximum(max_limit)
        editor.editingFinished.connect(self.editing_finished)
        return editor

    def setModelData(self, editor, model, index):
        new_float = editor.value()
        current_float = model.cell_data_by_index(index)
        if new_float == current_float:
            return
        model.setData(index, new_float, Qt.EditRole)
        if index.column() != 1:
            return
        _, line_index, param_name = model.dataframe().iloc[index.row()].name
        self.sigLineParamChanged.emit(new_float, line_index, param_name)
        self.RS.fitting.CommandDeconvLineDraggedAllowed = False
        command = CommandDeconvLineParameterChanged(self, self.RS, index, new_float, current_float, model,
                                                    line_index, param_name, "Edit line %s" % index)
        self.RS.undoStack.push(command)

    @pyqtSlot()
    def editing_finished(self):
        self.commitData.emit(self.sender())


class IntervalsTableDelegate(TableItemDelegate):

    def __init__(self, parent, rs):
        super().__init__(parent)
        self.rs = rs

    def createEditor(self, parent, option, index):
        editor = QDoubleSpinBox(parent)
        editor.setDecimals(5)
        editor.setSingleStep(1.)
        editor.setMinimum(0.)
        editor.setMaximum(10_000.)
        editor.editingFinished.connect(self.editing_finished)
        return editor

    def setModelData(self, editor, model, index):
        new_float = editor.value()
        current_float = model.cell_data_by_index(index)
        if new_float == current_float:
            return
        if index.column() != 0:
            return
        command = CommandFitIntervalChanged(self.rs, index, new_float, model, "Edit interval %s" % index)
        self.rs.undoStack.push(command)

    @pyqtSlot()
    def editing_finished(self):
        self.commitData.emit(self.sender())


class ComboDelegate(QStyledItemDelegate):
    sigLineTypeChanged = Signal(str, str, int)

    def __init__(self, line_types: set[str]):
        super().__init__()
        self.editorItems = line_types
        self.height = 30
        self.width = 140
        self._cursor_pos = QPoint(0, 0)
        self._cursor = QCursor()

    def createEditor(self, parent, option, index):
        editor = QListWidget(parent)
        self._cursor_pos = parent.mapFromGlobal(self._cursor.pos())
        editor.currentItemChanged.connect(self.current_item_changed)
        return editor

    def setEditorData(self, editor, index):
        z = 0
        for item in self.editorItems:
            ai = QListWidgetItem(item)
            editor.addItem(ai)
            if item == index.data():
                editor.setCurrentItem(editor.item(z))
            z += 1
        editor.setGeometry(self._cursor_pos.x(), self._cursor_pos.y(), self.width, self.height * len(self.editorItems))

    def setModelData(self, editor, model, index):
        new_text = editor.currentItem().text()
        current_text = model.cell_data_by_index(index)
        if current_text == new_text:
            return
        model.setData(index, new_text, Qt.EditRole)
        self.sigLineTypeChanged.emit(new_text, current_text, index.row())
        editor.close()

    @pyqtSlot()
    def current_item_changed(self):
        self.commitData.emit(self.sender())


