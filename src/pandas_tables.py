# pylint: disable=no-name-in-module, too-many-lines, invalid-name, import-error
# pylint: disable=too-many-public-methods

"""
Module for Qt interface with pandas dataframe.

Classes
-------
PandasModel
    Parent class for all other classes.
"""

import numpy as np
from PyQt5.QtCore import QAbstractItemModel
from PyQt5.QtWidgets import QStyleOptionViewItem, QWidget
from pandas import DataFrame, MultiIndex, concat, Series
from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal, Slot as pyqtSlot, QPoint
from qtpy.QtGui import QColor, QCursor
from qtpy.QtWidgets import QDoubleSpinBox, QStyledItemDelegate, QListWidget, QListWidgetItem

from qfluentwidgets import TableItemDelegate
from src import get_config
from src.backend.undo_stack import CommandUpdateTableCell, CommandFitIntervalChanged
from src.data.default_values import peak_shape_params_limits
from src.stages.fitting.classes.undo import CommandDeconvLineParameterChanged


class PandasModel(QAbstractTableModel):
    """
    A model to interface a Qt view with pandas dataframe.
    """

    def __init__(self, dataframe: DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """
        Override method from QAbstractTableModel.

        Returns
        -------
        out: int
            row count of the pandas DataFrame
        """
        if self._dataframe is not None and parent == QModelIndex() and not self._dataframe.empty:
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """
        Override method from QAbstractTableModel

        Returns
        -------
        out: int
            column count of the pandas DataFrame
        """
        if self._dataframe is None:
            return 0
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def clear_dataframe(self) -> None:
        """
        Remove all rows.
        """
        self._dataframe = self._dataframe[0:0]
        self.modelReset.emit()

    def data(self, index: QModelIndex, role=Qt.ItemDataRole) -> None | str | QColor:
        """
        Override method from QAbstractTableModel to render cell.

        Parameters
        -------
        index: QModelIndex
        role: Qt.ItemDataRole

        Returns
        -------
        data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]
        result = None
        match role:
            case Qt.DisplayRole:
                result = None if isinstance(value, QColor) else str(value)
            case Qt.ItemDataRole.EditRole:
                result = value
            case Qt.ItemDataRole.DecorationRole:
                result = value if isinstance(value, QColor) else None
            case Qt.ItemDataRole.ForegroundRole:
                if index.column() + 1 < self.columnCount():
                    value_color = self._dataframe.iloc[index.row(), index.column() + 1]
                    if isinstance(value_color, QColor):
                        result = value_color
            case _:
                return None
        return result

    def cell_data(self, row: int | str, column: int = 0):
        """
        Retrieve data from a specific cell in the dataframe.

        Parameters
        ----------
        row : int or str
            The row index (as an integer) or row label (as a string) from which to retrieve data.
        column : int, optional
            The column index from which to retrieve data, by default 0.

        Returns
        -------
        object
            The data stored in the specified cell of the dataframe.
        """
        if isinstance(row, int):
            result = self._dataframe.iloc[row, column]
        else:
            result = self._dataframe.at[row, column]
        return result

    def cell_data_by_idx_col_name(self, index: int, column_name: str) -> dict | None:
        """
        Retrieve data from a specific cell in the dataframe using an index and column name.

        Parameters
        ----------
        index : int
            The row index from which to retrieve data.
        column_name : str
            The name of the column from which to retrieve data.

        Returns
        -------
        dict or None
            The data stored in the specified cell of the dataframe as a dictionary.
            Returns None if the specified index or column name is not found.
        """
        try:
            return self._dataframe.loc[index, column_name]
        except KeyError:
            return None

    def set_cell_data_by_idx_col_name(self, index: int, column_name: str, value: str) -> None:
        """
        Set data in a specific cell in the dataframe using an index and column name.

        Parameters
        ----------
        index : int
            The row index where the data will be set.
        column_name : str
            The name of the column where the data will be set.
        value : str
            The value to set in the specified cell.

        Returns
        -------
        None
        """
        self._dataframe.loc[index, column_name] = value
        self.modelReset.emit()

    def cell_data_by_index(self, index: QModelIndex):
        """
        Retrieve data from a specific cell in the dataframe using a QModelIndex.

        Parameters
        ----------
        index : QModelIndex
            The QModelIndex object representing the cell location in the dataframe.

        Returns
        -------
        object
            The data stored in the specified cell of the dataframe.
        """
        return self._dataframe.iloc[index.row(), index.column()]

    def set_cell_data_by_index(self, index: QModelIndex, value: str) -> None:
        """
        Set data in a specific cell in the dataframe using a QModelIndex.

        Parameters
        ----------
        index : QModelIndex
            The QModelIndex object representing the cell location in the dataframe.
        value : str
            The value to set in the specified cell.

        Returns
        -------
        None

        Raises
        ------
        IndexError
            If the row or column index is out of bounds.
        """
        self._dataframe.iloc[index.row(), index.column()] = value
        self.modelReset.emit()

    def row_data(self, row: int) -> Series:
        """
        Retrieve data from a specific row in the dataframe.

        Parameters
        ----------
        row : int
            The row index from which to retrieve data.

        Returns
        -------
        pandas.Series
            The data stored in the specified row of the dataframe.

        Raises
        ------
        IndexError
            If the row index is out of bounds.
        """
        return self._dataframe.iloc[row]

    def row_data_by_filename(self, filename: str) -> Series:
        """
        Retrieve data from a specific row in the dataframe using a row label (filename).

        Parameters
        ----------
        filename : str
            The row label (filename) from which to retrieve data.

        Returns
        -------
        pandas.Series
            The data stored in the specified row of the dataframe.

        Raises
        ------
        KeyError
            If the row label (filename) is not found in the dataframe.
        """
        return self._dataframe.loc[filename]

    def get_df_by_multiindex(self, multi_index: MultiIndex) -> DataFrame | None:
        """
        Retrieve data from the dataframe using a MultiIndex.

        Parameters
        ----------
        multi_index : MultiIndex
            The MultiIndex object used to select data from the dataframe.

        Returns
        -------
        pandas.DataFrame or None
            A subset of the dataframe corresponding to the specified MultiIndex.
            Returns None if the MultiIndex does not match any rows in the dataframe.

        Raises
        ------
        KeyError
            If the MultiIndex is not found in the dataframe.
        """
        return self._dataframe.loc[multi_index]

    def query_result(self, q: str) -> DataFrame:
        """
        Execute a query on the dataframe and return the resulting subset.

        Parameters
        ----------
        q : str
            The query string to be executed on the dataframe.

        Returns
        -------
        pandas.DataFrame
            A new dataframe containing the rows that match the query conditions.
        """
        return self._dataframe.query(q)

    def column_data(self, column: int) -> Series:
        """
        Retrieve data from a specific column in the dataframe.

        Parameters
        ----------
        column : int
            The column index from which to retrieve data.

        Returns
        -------
        pandas.Series
            The data stored in the specified column of the dataframe.

        Raises
        ------
        IndexError
            If the column index is out of bounds.
        """
        return self._dataframe.iloc[:, column]

    def column_data_by_indexes(self, indexes: list, column: str) -> list:
        """
        Retrieve data from a specific column in the dataframe using a list of row indexes.

        Parameters
        ----------
        indexes : list
            A list of row indexes used to select data from the specified column.
        column : str
            The name of the column from which to retrieve data.

        Returns
        -------
        list
            A list containing the data from the specified column at the given row indexes.

        Raises
        ------
        KeyError
            If the column name is not found in the dataframe.
        IndexError
            If any of the provided indexes are out of bounds.
        """
        return list(self._dataframe[column][indexes])

    def get_column(self, column: str):
        """
        Retrieve data from a specific column in the dataframe by column name.

        Parameters
        ----------
        column : str
            The name of the column from which to retrieve data.

        Returns
        -------
        pandas.Series
            The data stored in the specified column of the dataframe.

        Raises
        ------
        KeyError
            If the column name is not found in the dataframe.
        """
        return self._dataframe[column]

    def append_dataframe(self, df2: DataFrame) -> None:
        """
        Append another DataFrame to the existing DataFrame and sort by index.

        Parameters
        ----------
        df2 : pandas.DataFrame
            The DataFrame to append to the existing DataFrame.

        Returns
        -------
        None

        Notes
        -----
        After appending `df2` to the existing DataFrame, the combined DataFrame is sorted by index.
        The `modelReset` signal is emitted after the operation.
        """
        self._dataframe = concat([self._dataframe, df2])
        self._dataframe = self._dataframe.sort_index()
        self.modelReset.emit()

    def get_group_by_name(self, name: str) -> int:
        """
        Retrieve the group value from a row identified by a given index name.

        Parameters
        ----------
        name : str
            The index name used to identify the row from which to retrieve the group value.

        Returns
        -------
        int
            The value from the third column of the row matching the given index name.
            Returns 0 if the index name is not found or the row is empty.

        Raises
        ------
        IndexError
            If the DataFrame does not contain at least three columns.
        """
        row = self._dataframe.loc[self._dataframe.index == name]
        if not row.empty:
            return row.values[0][2]
        return 0

    def change_cell_data(self, row: int | str, column: int | str,
                         value: QColor | float | dict) -> None:
        """
        Update the data in a specific cell of the dataframe.

        Parameters
        ----------
        row : int or str
            The row index (as an integer) or row label (as a string) where the data will be updated.
        column : int or str
            The column index (as an integer) or column label (as a string) where the data will be
            updated.
        value : QColor, float, or dict
            The new value to be set in the specified cell. Can be a QColor object, a float, or a
            dictionary.

        Returns
        -------
        None

        Notes
        -----
        If the row or column labels are not found in the DataFrame, an error may be raised.
        The `modelReset` signal is emitted after the data update to indicate that the model has been
        modified.
        """
        if isinstance(row, int):
            self._dataframe.iat[row, column] = value
        else:
            self._dataframe.at[row, column] = value
        self.modelReset.emit()

    def set_dataframe(self, df: DataFrame) -> None:
        """
        Set the dataframe to a new DataFrame and emit a reset signal.

        Parameters
        ----------
        df : pandas.DataFrame
            The new DataFrame to be set as the current dataframe. The DataFrame is copied to ensure
            that changes to the original DataFrame do not affect the internal dataframe.

        Returns
        -------
        None

        Notes
        -----
        The `modelReset` signal is emitted after updating the internal dataframe to indicate that
        the data model has been reset and any views or listeners should be updated.
        """
        self._dataframe = df.copy()
        self.modelReset.emit()

    @property
    def dataframe(self) -> DataFrame:
        """
        Get the current DataFrame.

        Returns
        -------
        pandas.DataFrame
            The DataFrame currently held by the instance.
        """
        return self._dataframe

    @property
    def first_index(self) -> int:
        """
        Get the first index value of the DataFrame.

        Returns
        -------
        int
            The value of the first index in the DataFrame.

        Raises
        ------
        IndexError
            If the DataFrame is empty and thus has no index.
        """
        return self._dataframe.index[0]

    def sort_index(self, ascending: bool = True) -> None:
        """
        Sort the DataFrame by its index.

        Parameters
        ----------
        ascending : bool, optional
            Whether to sort the index in ascending order. Default is True for ascending order.
            If False, the index will be sorted in descending order.

        Returns
        -------
        None

        Notes
        -----
        After sorting the DataFrame by index, the `modelReset` signal is emitted to indicate that
        the data model has been updated and any connected views or listeners should be refreshed.
        """
        self._dataframe = self._dataframe.sort_index(ascending=ascending)
        self.modelReset.emit()

    def reset_index(self) -> None:
        """
        Reset the index of the DataFrame, dropping the old index and replacing it with a default
        integer index.

        Returns
        -------
        None

        Notes
        -----
        The old index is dropped (i.e., not retained as a column) and replaced with a new default
        integer index starting from 0.
        After resetting the index, the `modelReset` signal is emitted to indicate that the data
        model has been updated and any connected views or listeners should be refreshed.
        """
        self._dataframe = self._dataframe.reset_index(drop=True)
        self.modelReset.emit()

    def setData(self, index: QModelIndex, value: QColor | float | dict, role: Qt.ItemDataRole) \
            -> bool:
        """
        Set the data for a specific cell in the DataFrame based on the provided index, value, and
        role.

        Parameters
        ----------
        index : QModelIndex
            The index of the cell where the data will be updated.
        value : QColor, float, or dict
            The new value to set in the cell. This can be a QColor object, a float, or a dictionary.
        role : Qt.ItemDataRole
            The role of the data to be set. Only data with `Qt.EditRole` will be updated.

        Returns
        -------
        bool
            Returns True if the data was successfully set; otherwise, returns False.

        Notes
        -----
        The method updates the cell value in the DataFrame only if the `role` is `Qt.EditRole` and
        the `value` is not an empty string.
        After updating the cell, the `dataChanged` signal is emitted to indicate that the data model
        has been modified.
        """
        if value != '' and role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = value
            self.dataChanged.emit(index, index)
            return True
        return False

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole) \
            -> str | None:
        """
        Retrieve header data for a specific section and orientation in the table model.

        This method overrides the `headerData` method from `QAbstractTableModel` to provide header
        data based on the DataFrame's columns and index.

        Parameters
        ----------
        section : int
            The section (column or row) for which the header data is requested.
        orientation : Qt.Orientation
            The orientation of the header. `Qt.Horizontal` for column headers and `Qt.Vertical` for
            row headers.
        role : Qt.ItemDataRole
            The role of the data being requested. This method returns data only for `Qt.DisplayRole`

        Returns
        -------
        str or None
            Returns the header data as a string if the role is `Qt.DisplayRole` and orientation is
            either horizontal or vertical. Returns `None` otherwise.

        Notes
        -----
        - For `Qt.Horizontal` orientation, the method returns the name of the column in the
            DataFrame corresponding to the `section`.
        - For `Qt.Vertical` orientation, the method returns the name of the index in the DataFrame
            corresponding to the `section`.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])
        return None

    def row_by_index(self, index: int) -> int | None:
        """
        Find the row position corresponding to a given index in the DataFrame.

        Parameters
        ----------
        index : int
            The index value to search for in the DataFrame's index.

        Returns
        -------
        int
            The row position (integer) of the given index in the DataFrame.
            Returns -1 if the index is not found.

        Notes
        -----
        - The method iterates through the DataFrame's index to find the position of the given index
            value.
        - If the index is not present, the method returns -1 to indicate that the index was not
            found.
        """
        for i, idx in enumerate(self._dataframe.index):
            if idx == index:
                return i
        return None

    def index_by_row(self, row: int) -> int:
        """
        Retrieve the index value corresponding to a specific row in the DataFrame.

        Parameters
        ----------
        row : int
            The row position in the DataFrame for which to retrieve the index value.

        Returns
        -------
        int
            The index value of the specified row. The return type is `int` if the index is
            integer-based; otherwise, it may be a different type depending on the DataFrame's index.

        Notes
        -----
        - The method uses `iloc` to access the row and retrieves its name (index value).
        - Assumes that the DataFrame has a unique index and that `row` is within the valid range of
            rows.
        """
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
        """
        Retrieve the index value corresponding to a specific value in a given column.

        Parameters
        ----------
        column_name : str
            The name of the column to search in.
        value :
            The value to search for in the specified column. The type of this parameter should match
            the type of values in the column.

        Returns
        -------
        int
            The index value of the row where the specified column has the given value. If multiple
            rows have the value, the index of the first matching row is returned.

        Raises
        ------
        IndexError
            If no rows match the specified value, an `IndexError` will be raised when trying to
            access the first element of an empty index.

        Notes
        -----
        - Assumes that the specified `column_name` exists in the DataFrame and that `value` is a
            valid value for comparison within that column.
        - This method returns the index of the first row where the specified column matches the
            given value.
        """
        return self._dataframe.loc[self._dataframe[column_name] == value].index[0]

    def set_column_data(self, col: str, ser: list):
        """
        Update a column in the DataFrame with new data.

        Parameters
        ----------
        col : str
            The name of the column to be updated in the DataFrame.
        ser : list
            A list of values to be assigned to the specified column. The length of this list must
            match the number of rows in the DataFrame.

        Returns
        -------
        None

        Notes
        -----
        - The method converts the list `ser` to a pandas `Series` and assigns it to the specified
            column `col`.
        - After updating the column data, the `modelReset` signal is emitted to indicate that the
            data model has been modified and any connected views or listeners should be refreshed.
        - It is assumed that the length of the list matches the number of rows in the DataFrame.
            If the lengths do not match, a `ValueError` will be raised by pandas.
        """
        self._dataframe[col] = Series(ser)
        self.modelReset.emit()

    def sort_values(self, current_name: str, ascending: bool) -> None:
        """
        Sort the DataFrame by the values of a specified column.

        Parameters
        ----------
        current_name : str
            The name of the column by which to sort the DataFrame.
        ascending : bool
            Whether to sort the column values in ascending order. If True, sorts in ascending order;
            if False, sorts in descending order.

        Returns
        -------
        None

        Notes
        -----
        - The method sorts the DataFrame based on the values in the column specified by
            `current_name`.
        - After sorting, the `modelReset` signal is emitted to indicate that the data model has
            been updated, and any connected views or listeners should be refreshed.
        - Assumes that the column name exists in the DataFrame. If the column does not exist, a
            `KeyError` will be raised by pandas.
        """
        self._dataframe = self._dataframe.sort_values(by=current_name, ascending=ascending)
        self.modelReset.emit()


class InputPandasTable(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for input_table
    """

    def __init__(self, dataframe: DataFrame):
        super().__init__(dataframe)
        self._dataframe = dataframe

    def data(self, index: QModelIndex, role=Qt.ItemDataRole) -> None | str:
        """
        Retrieve data for a specific cell from the DataFrame, based on the given index and role.

        Parameters
        ----------
        index : QModelIndex
            The index of the cell from which to retrieve data. This must be a valid index.
        role : Qt.ItemDataRole, optional
            The role of the data being requested. Only `Qt.DisplayRole` is supported for retrieving
            displayable data.
            Default is `Qt.ItemDataRole`.

        Returns
        -------
        None or str
            The data for the cell as a string if the role is `Qt.DisplayRole`, or `None` if the
            index is invalid or the role is not `Qt.DisplayRole`. Returns `None` if the data is of
            type `QColor`.

        Notes
        -----
        - This method is an override of the `data` method from `QAbstractTableModel`. It retrieves
            the value from the DataFrame based on the provided index and role.
        - For `Qt.DisplayRole`, the data is returned as a string representation unless it is an
            instance of `QColor`, in which case `None` is returned.
        - If the index is invalid, the method returns `None`.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            if isinstance(value, QColor):
                return None
            return str(value)
        return None

    @property
    def filenames(self) -> set:
        """
        Get the set of filenames from the DataFrame index.

        Returns
        -------
        set
            A set of filenames, which are the index values of the DataFrame. Each element in the set
            represents a unique index value.

        Notes
        -----
        - Assumes that the index of the DataFrame contains filenames or identifiers that can be
            retrieved as a set.
        - The returned set will include all unique index values present in the DataFrame.
        """
        return set(self._dataframe.index)

    def concat_df_input_table(self, index: list[str], col_data: dict) -> None:
        """
        Add rows to table.
        col_data must include keys like keys variable below.

        Parameters
        -------
        index: list[str]
            filenames
        col_data: dict
            lists for each key
        """
        keys = ['Min, nm', 'Max, nm', 'Group', 'Despiked, nm', 'Rayleigh line, nm', 'FWHM, nm',
                'FWHM, cm\N{superscript minus}\N{superscript one}', 'SNR']
        data = {}
        for k in keys:
            data[k] = [''] if k not in col_data else col_data[k]
        df2 = DataFrame(data, index=index)
        self._dataframe = concat([self._dataframe, df2])
        self._dataframe = self._dataframe.sort_index()
        self.modelReset.emit()

    def delete_rows(self, names: list[str]) -> None:
        """
        Delete rows from the DataFrame based on a list of index names.

        Parameters
        ----------
        names : list of str
            A list of index names (or labels) to be removed from the DataFrame. Each element in the
            list represents an index value of the rows to be deleted.

        Returns
        -------
        None

        Notes
        -----
        - The method removes all rows where the index matches any of the names provided in the list.
        - After deleting the rows, the DataFrame is sorted by its index.
        - The `modelReset` signal is emitted to notify that the data model has been updated, which
            may require refreshing any connected views or listeners.
        """
        self._dataframe = self._dataframe[~self._dataframe.index.isin(names)]
        self._dataframe = self._dataframe.sort_index()
        self.modelReset.emit()

    def get_filename_by_row(self, row: int) -> str:
        """
        Returns filename of N-th row.

        Parameters
        -------
        row: int
            0-indexed id

        Returns
        -------
        filename: str
            as index
        """
        filename = self._dataframe.index[row]
        return filename

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """
        Return the flags that specify the item properties for a given index.

        Parameters
        ----------
        index : QModelIndex
            The index of the item for which to retrieve the flags.

        Returns
        -------
        Qt.ItemFlags
            A bitwise OR combination of the flags that indicate the item properties.
            The flags determine if the item is selectable, enabled, and/or editable.

        Notes
        -----
        - For items in the third column (column index 2), the item is marked as selectable, enabled,
            and editable.
        - For all other columns, the item is only selectable and enabled.
        - This method is used by the model-view framework to determine the capabilities and
            interactions available for each item in the view.
        """
        if index.column() == 2:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def sort_values(self, current_name: str, ascending: bool) -> None:
        """
        Sort the DataFrame by the values of a specified column.

        Parameters
        ----------
        current_name : str
            The name of the column by which to sort the DataFrame.
        ascending : bool
            Whether to sort the column values in ascending order. If True, sorts in ascending order;
            if False, sorts in descending order.

        Returns
        -------
        None
            This method does not return a value.

        Notes
        -----
        - The DataFrame is sorted based on the values in the column specified by `current_name`.
        - After sorting, the DataFrame is updated and the `modelReset` signal is emitted to notify
          that the data model has been modified. This may require refreshing any connected views or
            listeners.
        - If the specified column does not exist in the DataFrame, a `KeyError` will be raised by
            pandas.
        """
        self._dataframe = self._dataframe.sort_values(by=current_name, ascending=ascending)
        self.modelReset.emit()

    def names_of_group(self, group_number: int) -> list[str]:
        """
        Retrieve the names of the rows belonging to a specific group.

        Parameters
        ----------
        group_number : int
            The group number used to filter the DataFrame. Only rows with this group number in the
            'Group' column will be considered.

        Returns
        -------
        list of str
            A list of index names (or labels) of the rows where the 'Group' column matches the
            specified group number.

        Notes
        -----
        - The method filters the DataFrame based on the 'Group' column and returns the index values
            of the filtered rows.
        - Assumes that the DataFrame has a column named 'Group' and that the column contains integer
            values representing group numbers.
        - The returned list will include all index names corresponding to the specified group number
        """
        return self._dataframe.loc[self._dataframe['Group'] == group_number].index

    def row_data_by_index(self, idx: str):
        """
        Retrieve a row of data from the DataFrame based on the given index label.

        Parameters
        ----------
        idx : str
            The index label used to locate the row in the DataFrame.

        Returns
        -------
        pandas.Series
            A Series containing the data of the row associated with the given index label.

        Notes
        -----
        - Assumes that the DataFrame's index contains string labels, and the provided `idx` matches
            one of these labels.
        - If the index label does not exist in the DataFrame, a `KeyError` will be raised.
        - The returned Series includes the data for the specified row, with the column names as the
            index.
        """
        return self._dataframe.loc[idx]

    @property
    def min_fwhm(self) -> float:
        """
        Compute the minimum value of the 'FWHM, cm⁻¹' column in the DataFrame.

        Returns
        -------
        float
            The minimum value found in the 'FWHM, cm⁻¹' column. This value represents the smallest
            Full Width at Half Maximum (FWHM) in the specified units.

        Notes
        -----
        - Assumes that the DataFrame contains a column named 'FWHM, cm⁻¹'. The column should contain
            numerical values.
        - If the column does not exist or is empty, the method will return an error. Ensure that the
            DataFrame contains valid numerical data in this column before accessing this property.
        - The column name 'FWHM, cm⁻¹' includes superscript characters for the minus sign.
        """
        return np.min(self.get_column('FWHM, cm\N{superscript minus}\N{superscript one}').values)

    def filenames_of_group_id(self, group_id: int) -> set[str]:
        """
        Returns filenames corresponding to group_id.

        Parameters
        -------
        group_id: int
            ID of group

        Returns
        -------
        filenames: set[str]
            of this group_id
        """
        rows = self._dataframe.loc[self._dataframe['Group'] == group_id]
        filenames = rows.index
        return set(filenames)


class GroupsTableModel(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for group table.
    """

    def __init__(self, dataframe: DataFrame, context):
        super().__init__(dataframe)
        self._dataframe = dataframe
        self.context = context

    def get_group_name_by_int(self, group_number: int) -> str:
        """
        Retrieve the group name corresponding to a specified group number.

        Parameters
        ----------
        group_number : int
            The group number used to find the corresponding group name.

        Returns
        -------
        str
            The name of the group associated with the specified group number.

        Notes
        -----
        - The method assumes that the DataFrame contains a column named 'Group name' and that the
            DataFrame’s index contains integer group numbers.
        - If there is no row in the DataFrame with the specified group number, or if the
            'Group name' column does not exist, the method may raise an error (e.g., `IndexError`
            or `KeyError`).
        - The method returns the group name from the first matching row found in the DataFrame.
        """
        return self._dataframe.loc[self._dataframe.index == group_number].iloc[0]['Group name']

    @property
    def groups_list(self) -> list:
        """
        Get the list of group labels from the DataFrame index.

        Returns
        -------
        list
            A list of index labels from the DataFrame. This represents the group labels or names
            associated with each row in the DataFrame.

        Notes
        -----
        - Assumes that the DataFrame's index contains the group labels or names.
        - The list returned includes all index labels present in the DataFrame.
        - If the DataFrame index is empty, an empty list will be returned.
        """
        return self._dataframe.index

    def append_group(self, group: str, style: dict, index: int) -> None:
        """
       Append a new group to the DataFrame with a specified style and index.

       Parameters
       ----------
       group : str
           The name of the group to be added to the DataFrame.
       style : dict
           A dictionary containing style information associated with the new group.
       index : int
           The index label for the new group entry. This index will be used for the row in the
            DataFrame.

       Returns
       -------
       None
           This method does not return a value.

       Notes
       -----
       - A new row with the provided `group` and `style` is appended to the DataFrame.
       - After appending, the DataFrame is sorted by its index and reset to have a continuous range
        of integer indices starting from 1.
       - The `modelReset` signal is emitted after updating the DataFrame to notify any connected
        views or listeners about the changes.
        """
        df2 = DataFrame({'Group name': [group],
                         'Style': [style]
                         }, index=[index])
        self._dataframe = concat([self._dataframe, df2])
        self._dataframe = self._dataframe.sort_index()
        self._dataframe = self._dataframe.reset_index(drop=True)
        self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self.modelReset.emit()

    def remove_group(self, row: int) -> None:
        """
        Remove a group from the DataFrame based on the specified row index.

        Parameters
        ----------
        row : int
            The index of the row to be removed from the DataFrame. This row corresponds to the group
            that will be deleted.

        Returns
        -------
        None
            This method does not return a value.

        Notes
        -----
        - The method drops the row with the specified index from the DataFrame.
        - After removing the row, the DataFrame is sorted by its index to maintain order.
        - The `modelReset` signal is emitted after updating the DataFrame to notify any connected
            views or listeners about the changes.
        - If the specified index does not exist in the DataFrame, a `KeyError` will be raised.
        """
        self._dataframe = self._dataframe.drop(row)
        self._dataframe = self._dataframe.sort_index()
        self.modelReset.emit()

    def flags(self, _: QModelIndex) -> Qt.ItemFlags:
        """
        Return the item flags for the given index in the model.

        Parameters
        ----------
        _ : QModelIndex
            The index for which the item flags are requested. This parameter is ignored in this
            implementation.

        Returns
        -------
        Qt.ItemFlags
            A bitwise OR of flags that specify the item’s properties. In this implementation, the
            item is selectable, enabled, and editable.

        Notes
        -----
        - This method is an override of a method from `QAbstractItemModel` and specifies the flags
            for items in the model.
        - The returned flags indicate that the item can be selected, is enabled, and is editable.
        - The `_` parameter is used as a placeholder to match the method signature expected by the
            model framework,
          but it is not used in this implementation.
        """
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Override method from QAbstractTableModel to return data from the DataFrame.

        Parameters
        ----------
        index : QModelIndex
            The index of the cell for which data is requested. This index identifies the row and
            column in the DataFrame.
        role : Qt.ItemDataRole, optional
            The role of the data being requested. Defines how the data should be presented. Common
            roles include
            Qt.DisplayRole, Qt.EditRole, Qt.DecorationRole, and Qt.ForegroundRole. The default is
            Qt.ItemDataRole.DisplayRole.

        Returns
        -------
        None | str | QColor
            The data for the specified role and index. The return type depends on the role:
            - `Qt.DisplayRole`: Returns a formatted string representation of the cell's value. If
            the value is a dictionary, it converts specific attributes into a string describing the
            line style and width.
            - `Qt.EditRole`: Returns a placeholder string 'changing...' for editable cells in the
            second column, or the original value for other cells.
            - `Qt.DecorationRole`: Returns a QColor object if the cell's value is a dictionary
                containing a color.
            - `Qt.ForegroundRole`: Returns a QColor object if the cell's value is in the adjacent
                column and is a QColor instance.

        Notes
        -----
        - This method handles different data roles by formatting or converting the cell's value
            accordingly.
        - The `Qt.DisplayRole` is used to provide a string representation of the cell's value, with
            special handling for dictionary values to format line styles and widths.
        - The `Qt.EditRole` provides a placeholder for cells in the second column to indicate
            editing state.
        - The `Qt.DecorationRole` and `Qt.ForegroundRole` return QColor objects if applicable,
            based on the data in the DataFrame.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]
        result = None
        match role:
            case Qt.DisplayRole:
                if isinstance(value, dict):
                    str_style = 'NoPen'
                    match value['style']:
                        case 1:
                            str_style = 'SolidLine'
                        case 2:
                            str_style = 'DashLine'
                        case 3:
                            str_style = 'DotLine'
                        case 4:
                            str_style = 'DashDotLine'
                        case 5:
                            str_style = 'DashDotDotLine'
                        case 6:
                            str_style = 'CustomDashLine'
                    result = f"{str_style}, {value['width']} pt"
                else:
                    result = str(value)
            case Qt.ItemDataRole.EditRole:
                result = 'changing...' if index.column() == 1 else value
            case Qt.ItemDataRole.DecorationRole:
                if isinstance(value, dict):
                    color = value['color']
                    color.setAlphaF(1.0)
                    result = color
            case Qt.ItemDataRole.ForegroundRole:
                if index.column() + 1 < self.columnCount():
                    value_color = self._dataframe.iloc[index.row(), index.column() + 1]
                    if isinstance(value_color, QColor):
                        result = value_color
        return result

    def setData(self, index, value, role):
        """
        Override method from QAbstractTableModel to set data in the DataFrame and manage undo
        commands.

        Parameters
        ----------
        index : QModelIndex
            The index of the cell where data should be set. This index identifies the row and column
            in the DataFrame.
        value : QColor | float | dict
            The new value to be set in the cell. The type of `value` depends on the column's data
            type.
        role : Qt.ItemDataRole
            The role of the data being set. Determines how the data is processed and stored. Common
            roles include Qt.EditRole.

        Returns
        -------
        bool
            `True` if the data was successfully set and an undo command was created, otherwise
            `False`.

        Notes
        -----
        - The method handles only the `Qt.EditRole`. If the role is `Qt.EditRole` and the index is
            not in the second column (column index 1), it creates an `undo` command for updating the
            cell value.
        - An `undo` command is created using `CommandUpdateTableCell`, which allows changes to be
            undone if needed.
        - The command is pushed onto the undo stack provided by the `context` for undo
            functionality.
        - If the role is not `Qt.EditRole` or if the index is in the second column, the method
            returns `False` indicating no action was taken.
        """
        if role == Qt.EditRole and not index.column() == 1:
            old_value = self._dataframe.iloc[index.row(), index.column()]
            command = CommandUpdateTableCell((value, old_value), self.context,
                                             "Change group name", **{'index': index,
                                                                     'obj': self})
            self.context.undo_stack.push(command)
            return True
        return False

    @property
    def groups_styles(self) -> list:
        """
        Get the list of group styles from the DataFrame.

        Returns
        -------
        list
            A list containing the styles from the second column of the DataFrame.

        Notes
        -----
        - This property extracts the data from the second column (index 1) of the DataFrame and
            converts it into a list.
        - The returned list contains the styles associated with each group in the DataFrame.
        """
        return list(self._dataframe.iloc[:, 1])

    @property
    def groups_colors(self) -> list:
        """
        Get the list of colors associated with each group from the DataFrame.

        Returns
        -------
        list
            A list of color names corresponding to the styles in the second column of the DataFrame.

        Notes
        -----
        - This property extracts the 'color' attribute from each style dictionary in the
            q`groups_styles` property.
        - Each color is converted to its name representation using the `name()` method of the
            `QColor` class.
        - The resulting list contains the color names in the same order as the styles in the
            DataFrame.
        """
        colors = []
        for style in self.groups_styles:
            colors.append(style['color'].name())
        return colors

    @property
    def groups_width(self) -> list:
        """
        Get the list of widths associated with each group from the DataFrame.

        Returns
        -------
        list
            A list of widths corresponding to the styles in the second column of the DataFrame.

        Notes
        -----
        - This property extracts the 'width' attribute from each style dictionary in the
            `groups_styles` property.
        - The resulting list contains the widths in the same order as the styles in the DataFrame.
        """
        width = []
        for style in self.groups_styles:
            width.append(style['width'])
        return width

    @property
    def groups_dashes(self) -> list:
        """
        Get the list of dash patterns associated with each group from the DataFrame.

        Returns
        -------
        list
            A list of dash patterns corresponding to the styles in the second column of the
            DataFrame.
            Each pattern is represented as a tuple of lengths or an empty string for solid lines.

        Notes
        -----
        - This property extracts the 'style' attribute from each style dictionary in the
            `groups_styles` property.
        - It converts each `Qt.PenStyle` value into a corresponding dash pattern:
            - `Qt.PenStyle.SolidLine` results in an empty string (no dashes).
            - `Qt.PenStyle.DotLine` results in a tuple `(1, 1)`.
            - `Qt.PenStyle.DashLine` results in a tuple `(4, 4)`.
            - `Qt.PenStyle.DashDotLine` results in a tuple `(4, 1.25, 1.5, 1.25)`.
            - `Qt.PenStyle.DashDotDotLine` results in a tuple `(4, 1.25, 1.5, 1.25, 1.5, 1.25)`.
        - The resulting list contains the dash patterns in the same order as the styles in the
            DataFrame.
        """
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
    """
    A model to interface a Qt view with pandas dataframe for decomposition filenames table.
    """

    def __init__(self, dataframe: DataFrame):
        super().__init__(dataframe)
        self._dataframe = dataframe

    def concat_deconv_table(self, filename: list[str]) -> None:
        """
        Add rows to table.

        Parameters
        -------
        filename: list[str]
            filenames
        """
        df2 = DataFrame({'Filename': filename})
        self._dataframe = concat([self._dataframe, df2])
        self._dataframe = self._dataframe.sort_values(by=['Filename'])
        self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self.modelReset.emit()

    def delete_rows(self, names: list[str]) -> None:
        """
        Delete rows from the DataFrame where the 'Filename' column matches any of the given names.

        Parameters
        ----------
        names : list of str
            A list of filenames. Rows with 'Filename' values matching any of these names will be
            removed from the DataFrame.

        Returns
        -------
        None
            This method does not return any value. It modifies the DataFrame in place.

        Notes
        -----
        - The method filters out rows from the DataFrame where the 'Filename' column matches any
            value in the `names` list.
        - After deleting the rows, the DataFrame is sorted by the 'Filename' column in ascending
            order.
        - The DataFrame's index is reset to be a sequential range starting from 1.
        - The `modelReset` signal is emitted to notify any views or listeners that the model data
            has changed.
        """
        self._dataframe = self._dataframe[~self._dataframe.Filename.isin(names)]
        self._dataframe = self._dataframe.sort_values(by=['Filename'])
        self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self.modelReset.emit()

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Override method from QAbstractTableModel

        Retrieve the data for the given index from the pandas DataFrame.

        Parameters
        ----------
        index : QModelIndex
            The index specifying the row and column for which data is to be retrieved.
        role : Qt.ItemDataRole, optional
            The role for which data is requested. Default is `Qt.DisplayRole`.

        Returns
        -------
        None | str
            The data for the specified index and role. Returns `None` if the index is invalid or
                the role is not supported.
            - If `role` is `Qt.DisplayRole` and the value is a `QColor`, `None` is returned.
            - Otherwise, the value is converted to a string.

        Notes
        -----
        - The `data` method handles different roles to provide appropriate data from the DataFrame.
        - For `Qt.DisplayRole`, it converts the DataFrame value to a string, except when the value
            is a `QColor`, in which case it returns `None`.
        - For roles other than `Qt.DisplayRole`, `None` is returned, as other roles are not
            supported in this implementation.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            if isinstance(value, QColor):
                return None
            return str(value)
        return None


class PandasModelFitIntervals(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for fit intervals table.
    """

    def __init__(self, dataframe: DataFrame):
        super().__init__(dataframe)
        self._dataframe = dataframe

    def append_row(self) -> None:
        """
        Append a new row with a default value to the DataFrame and sort by the 'Border' column.

        Returns
        -------
        None
            This method does not return any value. It modifies the DataFrame in place.

        Notes
        -----
        - A new row with the value `0.` for the 'Border' column is appended to the existing
            DataFrame.
        - After appending the row, the DataFrame is sorted based on the 'Border' column in ascending
            order.
        - This method assumes that the DataFrame has a 'Border' column and that `sort_by_border`
            is a method that sorts the DataFrame by this column.
        """
        df2 = DataFrame({'Border': [0.]})
        self._dataframe = concat([self._dataframe, df2])
        self.sort_by_border()

    def delete_row(self, interval_number: int) -> None:
        """
        Delete a row from the DataFrame based on the given index and sort the DataFrame by the
        'Border' column.

        Parameters
        ----------
        interval_number : int
            The index of the row to be deleted from the DataFrame.

        Returns
        -------
        None
            This method does not return any value. It modifies the DataFrame in place.

        Notes
        -----
        - The method removes the row corresponding to `interval_number` from the DataFrame.
        - After deleting the row, the DataFrame is sorted by the 'Border' column using the
            `sort_by_border` method.
        - It is assumed that the `sort_by_border` method handles the sorting of the DataFrame by the
            'Border' column.
        """
        self._dataframe = self._dataframe.drop(interval_number)
        self.sort_by_border()

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Override method from QAbstractTableModel to return data from the pandas DataFrame based on
        the role.

        Parameters
        ----------
        index : QModelIndex
            The index specifying the row and column of the data to retrieve.
        role : Qt.ItemDataRole, optional
            The role for which data is requested. Defaults to `Qt.DisplayRole`.

        Returns
        -------
        None | str | float
            The data for the specified index and role. The return type depends on the role:
            - For `Qt.DisplayRole`, the data is returned as a string.
            - For `Qt.EditRole`, the data is returned as a float.
            - Returns `None` for other roles or if the index is invalid.

        Notes
        -----
        - The method handles data retrieval for different roles as follows:
            - `Qt.DisplayRole`: Converts the DataFrame value to a string.
            - `Qt.EditRole`: Converts the DataFrame value to a float, useful for editing purposes.
        - If the index is invalid or the role is not supported, `None` is returned.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.EditRole:
            return float(value)
        return None

    def flags(self, _: QModelIndex) -> Qt.ItemFlags:
        """
        Override method from QAbstractItemModel to return item flags for the specified index.

        Parameters
        ----------
        _ : QModelIndex
            The index for which the item flags are requested.

        Returns
        -------
        Qt.ItemFlags
            The flags indicating the capabilities of the item at the specified index.
            - `Qt.ItemIsSelectable`: The item can be selected.
            - `Qt.ItemIsEnabled`: The item is enabled.
            - `Qt.ItemIsEditable`: The item can be edited.

        Notes
        -----
        - This method specifies that all items in the model are selectable, enabled, and editable.
        - This implementation assumes that all items in the model share the same properties.
        """
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def sort_by_border(self) -> None:
        """
        Sort the DataFrame by the 'Border' column and reset the index.

        This method sorts the DataFrame in ascending order based on the values in the 'Border'
        column.
        After sorting, the DataFrame's index is reset to be a sequential range starting from 1.
        The `modelReset` signal is emitted to notify any views or components that the model has been
        updated.

        Returns
        -------
        None
            This method does not return any value. It modifies the DataFrame in place.

        Notes
        -----
        - The method assumes that the DataFrame contains a 'Border' column. If the column does not
            exist, this will raise an error.
        - The index is reset to start from 1 and increments sequentially. This is done to ensure the
            DataFrame has a clean, continuous index after sorting.
        - The `modelReset` signal is emitted to notify that the model data has changed, which is
            useful for updating views in a UI.
        """
        self._dataframe = self._dataframe.sort_values(by=['Border'])
        self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self.modelReset.emit()

    def add_auto_found_borders(self, borders: np.ndarray) -> None:
        """
        Add new border values to the DataFrame and reset the model.

        This method updates the DataFrame with new border values provided as a NumPy array.
        It replaces the existing DataFrame content with a new DataFrame where the 'Border' column
        is populated with the values from the `borders` array. After updating, the `modelReset`
        signal is emitted to notify that the model data has been refreshed.

        Parameters
        ----------
        borders : np.ndarray
            A NumPy array containing border values to be added to the DataFrame. The array should
            be one-dimensional.

        Returns
        -------
        None
            This method does not return any value. It modifies the DataFrame in place.

        Notes
        -----
        - The method assumes that `borders` is a one-dimensional NumPy array. If it is
            multidimensional, this will raise an error.
        - The DataFrame is completely replaced with a new DataFrame containing only the 'Border'
            column.
        - The `modelReset` signal is emitted to indicate that the model data has been updated.
        """
        self._dataframe = DataFrame({'Border': borders})
        self.modelReset.emit()


class PandasModelDeconvLinesTable(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for table with decomposition lines.
    """

    sigCheckedChanged = Signal(int, bool)

    def __init__(self, context, dataframe: DataFrame, checked: list[bool]):
        super().__init__(dataframe)
        self.context = context
        self._dataframe = dataframe
        self._checked = checked

    def append_row(self, legend: str = 'Curve 1', line_type: str = 'Gaussian', style=None,
                   idx: int = -1) -> int:
        """
        Append a new row to the DataFrame with the specified attributes.

        This method adds a new row to the DataFrame with the given legend, line type, and style.
        If an index is not provided, a new index is automatically assigned. After appending the row,
        the `modelReset` signal is emitted to notify any  views or components that the model data
        has been updated. The method returns the index of the newly added row.

        Parameters
        ----------
        legend : str, optional
            The legend for the new row. Defaults to 'Curve 1'.
        line_type : str, optional
            The type of line for the new row. Defaults to 'Gaussian'.
        style : dict, optional
            A dictionary specifying the style attributes for the new row. Defaults to an empty
            dictionary if not provided.
        idx : int, optional
            The index at which to insert the new row. If -1, a new index is assigned. Defaults to -1

        Returns
        -------
        int
            The index of the newly added row.

        Notes
        -----
        - The method assumes that `style` is a dictionary. If it is not provided, an empty
            dictionary is used.
        - If `idx` is -1, a free index is automatically determined by the `free_index` method.
        - The `_checked` list is updated to include a new `True` value for the newly added row.
        - The `modelReset` signal is emitted to notify that the model data has been updated.
        """
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
        """
        Get the index of the DataFrame.

        This property returns the index of the DataFrame. The index is a list of labels that
        identify the rows in the DataFrame.

        Returns
        -------
        pandas.Index
            The index of the DataFrame, which is a sequence of labels for the rows.
        """
        return self._dataframe.index

    def free_index(self) -> int:
        """
        Determine the next available index that is not currently used in the DataFrame.

        This method finds the smallest integer greater than or equal to the current length of the
        DataFrame that is not already present in the DataFrame's index. It ensures that the index
        returned is unique and does not conflict with existing indices.

        Returns
        -------
        int
            The next available index that is not currently used in the DataFrame.

        Notes
        -----
        - The method starts by assuming the next index to be the length of the DataFrame.
        - If the assumed index is already present in the DataFrame's index, it increments the index
          until an available index is found.
        - The method is useful for ensuring unique indices when adding new rows to the DataFrame.
        """
        idx = len(self._dataframe)
        if idx in self._dataframe.index:
            while idx in self._dataframe.index:
                idx += 1
        return idx

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Override method from QAbstractTableModel to return data from the pandas DataFrame.

        This method provides data for a specific cell in the DataFrame based on the given role.
        It handles different roles such as display, edit, decoration, and check state, transforming
        the cell's value accordingly.

        Parameters
        ----------
        index : QModelIndex
          The index of the cell whose data is to be retrieved.
        role : Qt.ItemDataRole, optional
          The role of the data to be retrieved. Default is `Qt.ItemDataRole`.

        Returns
        -------
        None, str, QColor, Qt.AlignmentFlag, Qt.CheckState
          - `None` if the index is invalid or the role does not match.
          - `str` if the role is `Qt.DisplayRole`, providing a formatted string for display.
          - `QColor` if the role is `Qt.DecorationRole`, providing a color for decoration.
          - `Qt.AlignmentFlag` if the role is `Qt.TextAlignmentRole`, specifying the alignment.
          - `Qt.CheckState` if the role is `Qt.CheckStateRole`, indicating the checked state.

        Notes
        -----
        - `Qt.DisplayRole` converts a dictionary's style and width information to a readable string.
        - `Qt.EditRole` returns `'changing...'` for a specific column; otherwise, it returns the
            cell value.
        - `Qt.DecorationRole` provides a `QColor` with full opacity if the cell contains a color
            dictionary.
        - `Qt.TextAlignmentRole` sets the vertical center and left alignment.
        - `Qt.CheckStateRole` returns the checked state for rows if the column is 0, using the
            `_checked` list.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]
        result = None
        match role:
            case Qt.DisplayRole:
                if isinstance(value, dict):
                    str_style = 'NoPen'
                    match value['style']:
                        case 1:
                            str_style = 'SolidLine'
                        case 2:
                            str_style = 'DashLine'
                        case 3:
                            str_style = 'DotLine'
                        case 4:
                            str_style = 'DashDotLine'
                        case 5:
                            str_style = 'DashDotDotLine'
                        case 6:
                            str_style = 'CustomDashLine'
                    result = f"{str_style}, {value['width']} pt"
                else:
                    result = str(value)
            case Qt.ItemDataRole.EditRole:
                result = 'changing...' if index.column() == 1 else value
            case Qt.ItemDataRole.DecorationRole:
                if isinstance(value, dict):
                    color = value['color']
                    color.setAlphaF(1.0)
                    result = color
            case Qt.TextAlignmentRole:
                result = Qt.AlignVCenter + Qt.AlignmentFlag.AlignLeft
            case Qt.CheckStateRole:
                if index.column() == 0:
                    checked_row = self._dataframe.index[index.row()]
                    checked = self._checked[checked_row]
                    result = Qt.Checked if checked else Qt.Unchecked
        return result

    def flags(self, _: QModelIndex) -> Qt.ItemFlags:
        """
        Override method from QAbstractItemModel to return item flags.

        This method specifies the item flags for each cell in the model. The flags indicate the
        capabilities and interactions available for the items, such as whether they can be selected,
        edited, or checked.

        Parameters
        ----------
        _ : QModelIndex
            The index of the item for which the flags are being queried. This parameter is not used
            in the method but is required for compatibility with the QAbstractItemModel interface.

        Returns
        -------
        Qt.ItemFlags
            A combination of flags indicating the item's capabilities. The flags include:
            - `Qt.ItemIsSelectable`: The item can be selected.
            - `Qt.ItemIsEnabled`: The item is enabled and can be interacted with.
            - `Qt.ItemIsEditable`: The item can be edited.
            - `Qt.ItemIsUserCheckable`: The item can be checked or unchecked by the user.

        Notes
        -----
        - The method combines several flags to indicate that the item is selectable, enabled,
            editable, and user-checkable.
        - This method helps the Qt framework determine how the item should be displayed and
            interacted with in the view.
        """
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsUserCheckable

    def setData(self, index, value, role):
        """
        Override method from QAbstractItemModel to set data for a specific item.

        This method updates the data of the item at the specified index based on the given role.
        It handles changes in cell data for editing and check state roles.

        Parameters
        ----------
        index : QModelIndex
            The index of the item to be updated.
        value : any
            The new value to set in the data model. The type depends on the role:
            - For `Qt.EditRole`, it is the new data value.
            - For `Qt.CheckStateRole`, it is the check state value.
        role : Qt.ItemDataRole
            The role that specifies the type of data being set. This method handles the following
                roles:
            - `Qt.EditRole`: Used for editing data in the model.
            - `Qt.CheckStateRole`: Used for setting the check state of an item.

        Returns
        -------
        bool
            Returns `True` if the data was successfully set, otherwise `False`.

        Notes
        -----
        - If the `Qt.EditRole` is used and the column is 0, it updates the cell data and creates an
            undo command for the change.
        - If the `Qt.CheckStateRole` is used, it updates the check state of the item and emits a
            signal to notify about the change.
        - For other roles, or if the operation fails, it returns `False`.
        """
        if role == Qt.EditRole and index.column() == 0:
            old_value = self._dataframe.iloc[index.row(), index.column()]
            command = CommandUpdateTableCell((value, old_value), self.context,
                                             "Change line name", **{'index': index,
                                                                    'obj': self})
            self.context.undo_stack.push(command)
            return True
        if role == Qt.CheckStateRole:
            checked = value == 2
            checked_idx = self._dataframe.iloc[index.row()].name
            self._checked[checked_idx] = checked
            self.sigCheckedChanged.emit(checked_idx, checked)
            return True
        return False

    def sort_values(self, current_name: str, ascending: bool) -> None:
        """
        Sort the DataFrame by a specified column.

        This method sorts the DataFrame based on the values in the specified column. It updates the
        DataFrame and emits a `modelReset` signal to notify views of the change.

        Parameters
        ----------
        current_name : str
            The name of the column by which to sort the DataFrame. The column must exist in the
            DataFrame.
        ascending : bool
            A boolean flag indicating the sort order. If `True`, the DataFrame is sorted in
            ascending order; if `False`, it is sorted in descending order.

        Returns
        -------
        None
            This method does not return a value. It modifies the DataFrame in place.

        Notes
        -----
        - The method uses the `sort_values` function from pandas to perform the sorting operation.
        - After sorting, the `modelReset` signal is emitted to ensure that any views or models
            connected to this data are updated to reflect the new order.
        """
        self._dataframe = self._dataframe.sort_values(by=current_name, ascending=ascending)
        self.modelReset.emit()

    def checked(self) -> list[bool]:
        """
        Retrieve the list of check states.

        This method returns a list indicating the checked state of each item. Each entry in the list
        is a boolean value where `True` represents a checked item and `False` represents an
        unchecked item.

        Returns
        -------
        list[bool]
            A list of boolean values where each boolean indicates whether the corresponding item is
            checked (`True`) or unchecked (`False`).

        Notes
        -----
        - The list corresponds to the check states of items in the DataFrame, which is managed
            internally by the class.
        - The order of the list matches the order of items in the DataFrame.
        """
        return self._checked

    def clear_dataframe(self) -> None:
        """
        Clear the DataFrame and reset check states.

        This method clears the DataFrame by retaining its structure but removing all rows. It also
        resets the list of checked states to an empty list. After performing these actions, it emits
        the `modelReset` signal to notify any connected views or models of the update.

        Returns
        -------
        None
            This method does not return a value. It modifies the DataFrame and checked states in
            place.

        Notes
        -----
        - The DataFrame is cleared by slicing it with `[0:0]`, which retains the column structure
            but removes all rows.
        - The `_checked` attribute is reset to an empty list, which should match the cleared
            DataFrame's row count.
        - The `modelReset` signal is emitted to update any views or models that are observing the
            DataFrame.
        """
        self._dataframe = self._dataframe[0:0]
        self._checked = []
        self.modelReset.emit()

    def set_checked(self, checked: list[bool]) -> None:
        """
        Set the checked states for items.

        This method updates the internal list of checked states with the provided list. Each boolean
        in the list represents the checked state of an item, where `True` indicates that the item is
        checked, and `False` indicates that it is unchecked.

        Parameters
        ----------
        checked : list[bool]
            A list of boolean values where each boolean indicates whether the corresponding item is
            checked (`True`) or unchecked (`False`). The length of the list should match the number
            of items in the DataFrame.

        Returns
        -------
        None
            This method does not return a value. It updates the internal state in place.

        Notes
        -----
        - The length of the `checked` list must match the number of rows in the DataFrame. If the
            length does not match, it may lead to inconsistent states.
        - The `modelReset` signal is not emitted by this method. If the DataFrame needs to reflect
            these changes, ensure that the appropriate signal is emitted elsewhere if necessary.
        """
        self._checked = checked

    def delete_row(self, idx: int) -> None:
        """
        Delete a row from the DataFrame.

        This method removes the row with the specified index from the DataFrame. After the row is
        deleted, it  emits the `modelReset` signal to notify any connected views or models of the
        change.

        Parameters
        ----------
        idx : int
            The index of the row to be deleted from the DataFrame. This index must be present in the
            DataFrame's index.

        Returns
        -------
        None
            This method does not return a value. It modifies the DataFrame in place and emits a
            signal to update any connected views or models.

        Notes
        -----
        - If the specified index does not exist in the DataFrame, the method will raise a `KeyError`
        - The `modelReset` signal is emitted to update any views or models observing the DataFrame
            to reflect the changes.
        """
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


class PandasModelFitParamsTable(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for table with fit params.
    """

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    @property
    def filenames(self) -> list[str]:
        """
        Get the list of filenames from the DataFrame index.

        This property retrieves the first level of the DataFrame's MultiIndex, which is expected to
        contain filenames.
        It returns a list of filenames, filtering out any empty values.

        Returns
        -------
        list[str]
            A list of filenames extracted from the first level of the DataFrame's MultiIndex. Empty
            values are filtered out.

        Notes
        -----
        - The DataFrame is expected to have a MultiIndex with filenames at the first level.
        - If the DataFrame does not use a MultiIndex or if the first level does not contain
            filenames, the method may return an empty list or raise an error.
        - Ensure that the DataFrame's index structure matches the expected format for correct
            operation.
        """
        filenames = self._dataframe.index.levels[0].values
        filenames = [i for i in filenames if i]
        return filenames

    def batch_unfitted(self) -> bool:
        """
        Returns False if was fitted any spectrum except template.
        True - no one was fitted
        """
        filenames = np.unique(self._dataframe.index.get_level_values(0))
        return filenames.size == 1 and filenames[0] == ''

    def append_row(self, line_index: int, param_name: str, param_value: float, min_v: float = None,
                   max_v: float = None, filename: str = '') -> None:
        """
        Append a new row to the DataFrame with fitting parameters for a specific line.

        This method appends a new row to the DataFrame, which includes parameters for a fitting line
        The method sets default limits for the parameter values based on predefined rules and
        configuration settings. It updates the  DataFrame with the new parameter values and triggers
        a model reset.

        Parameters
        ----------
        line_index : int
            The index of the line to which the parameter is associated.
        param_name : str
            The name of the parameter being added.
        param_value : float
            The value of the parameter being added.
        min_v : float, optional
            The minimum value for the parameter. If not provided, a default value based on
            `param_name` and configuration settings is used.
        max_v : float, optional
            The maximum value for the parameter. If not provided, a default value based on
            `param_name` and configuration settings is used.
        filename : str, optional
            An optional filename to associate with the parameter. Defaults to an empty string.

        Returns
        -------
        None
            This method does not return any value. It updates the DataFrame in place.

        Notes
        -----
        - Parameter limits for 'a', 'x0', and 'dx' are computed based on predefined rules.
            Other parameters use limits specified in the configuration.
        - If `min_v` or `max_v` are provided, they will override the computed limits for the
            parameter.
        """
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

    def set_parameter_value(self, filename: str, line_index: int, parameter_name: str,
                            column_name: str, value: float, emit: bool = True) -> None:
        """
        Set value of current cell.

        Parameters
        ---------
        filename : str
            spectrum filename - 0 idx in MultiIndex

        line_index : int
            curve index - 1 idx in MultiIndex

        parameter_name : str
            parameter name - 2 idx in MultiIndex ('a', x0, dx, etc.)

        column_name : str
            Value, Min value or Max value

        value : float
            Value to set

        emit : bool
            emit changes or not

        """
        self._dataframe.loc[(filename, line_index, parameter_name), column_name] = np.round(value,
                                                                                            5)
        if emit:
            self.modelReset.emit()

    def model_reset_emit(self) -> None:
        """
        Emit the `modelReset` signal.

        This method triggers the `modelReset` signal, which is typically used to notify views or
        other components that the underlying model has been reset and requires a refresh. It does
        not modify the DataFrame or other attributes but ensures that any connected views or
        listeners are updated accordingly.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        - The `modelReset` signal is a common signal used in Qt-based models to indicate that the
            data model has been entirely reset. This is often used to update views that depend on
            the model's data.
        """
        self.modelReset.emit()

    def get_parameter_value(self, filename: str, line_index: int, parameter_name: str,
                            column_name: str) -> float:
        """
        Retrieve the value of a specific parameter from the DataFrame.

        This method extracts the value of a given parameter from the DataFrame based on the
        provided indices and column name. The indices used are `filename`, `line_index`, and
        `parameter_name`, while the  column name specifies which column to retrieve the value from.

        Parameters
        ----------
        filename : str
            The filename associated with the data entry.
        line_index : int
            The line index within the specified filename.
        parameter_name : str
            The name of the parameter whose value is to be retrieved.
        column_name : str
            The name of the column from which to retrieve the parameter value.

        Returns
        -------
        float
            The value of the specified parameter from the specified column.

        Notes
        -----
        - The method assumes that the DataFrame has a MultiIndex with levels: 'filename',
            'line_index', and 'parameter_name', and that `column_name` is a valid column in the
            DataFrame.
        - If the specified indices or column name do not exist, this method may raise a KeyError or
            IndexError.
        """
        return self._dataframe.loc[(filename, line_index, parameter_name), column_name]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """
        Override method from QAbstractItemModel to return item flags for the given index.

        This method determines the item flags for a specific cell in the model, which define the
        cell's interaction capabilities, such as whether it is selectable, enabled, or editable.
        The flags vary depending on the column of the item.

        Parameters
        ----------
        index : QModelIndex
            The index of the item for which the flags are being retrieved.

        Returns
        -------
        Qt.ItemFlags
            The flags that indicate the item's interaction capabilities.

        Notes
        -----
        - If the column index is 0, the item will be selectable and enabled, but not editable.
        - For all other columns, the item will be selectable, enabled, and editable.
        - The returned flags are a combination of `Qt.ItemFlags` values using bitwise OR (`|`).

        See Also
        --------
        Qt.ItemFlags
            An enumeration of possible item flags, which control the behavior and appearance of
            items in the model.
        """
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Override method from QAbstractTableModel to return data for the given index and role.

        This method retrieves the data from the pandas DataFrame based on the role requested by the
        view.
        It handles different roles such as display and edit roles to return the appropriate data
        format.

        Parameters
        ----------
        index : QModelIndex
            The index of the item in the model whose data is to be retrieved.

        role : Qt.ItemDataRole, optional
            The role for which the data is requested. Default is `Qt.ItemDataRole.DisplayRole`.

        Returns
        -------
        None | str | float
            - If `role` is `Qt.DisplayRole`, returns the data as a string.
            - If `role` is `Qt.EditRole`, returns the data as a float. The value is used for editing
                purposes.
            - Returns `None` for other roles or if the index is invalid.

        Notes
        -----
        - `Qt.DisplayRole` is used to display data in the view, so the data is converted to a string
            for display.
        - `Qt.EditRole` is used for editing data. Although a `QDoubleSpinBox` is mentioned in the
            code, it is not used directly in this method. The data is simply returned as a float.
        - If the index is invalid or the role is not handled, `None` is returned.

        See Also
        --------
        QAbstractItemModel.data : Base method from which this method is overridden.
        Qt.ItemDataRole : Enum for different data roles such as display and edit roles.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.EditRole:
            double_spinbox = QDoubleSpinBox()
            double_spinbox.setValue(float(value))
            double_spinbox.setDecimals(5)
            return float(value)
        return None

    def setData(self, index, value, role):
        """
        Override method from QAbstractTableModel to set data for the given index and role.

        This method updates the data in the pandas DataFrame based on the role requested by the view
        It handles the editing role to modify the data within the model and emits the `dataChanged`
        signal to notify of the update.

        Parameters
        ----------
        index : QModelIndex
            The index of the item in the model whose data is to be set.

        value : float
            The new value to set for the specified index.

        role : Qt.ItemDataRole
            The role for which the data is being set. Must be `Qt.EditRole` for the data to be
            updated.

        Returns
        -------
        bool
            Returns `True` if the data was successfully set, `False` otherwise.

        Notes
        -----
        - The method only updates the data if the role is `Qt.EditRole`. For other roles, it returns
            `False`.
        - After updating the data, the `dataChanged` signal is emitted to notify any views or
            proxies about the data change.

        See Also
        --------
        QAbstractItemModel.setData : Base method from which this method is overridden.
        Qt.ItemDataRole : Enum for different data roles including edit roles.
        """
        if role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True
        return False

    def delete_rows(self, idx: int, level: int = 1) -> None:
        """
        Delete rows from the DataFrame based on the given index and level in the MultiIndex.

        This method removes rows from the DataFrame where the index matches the specified `idx` at
        the given `level`.
        After removing the rows, the `modelReset` signal is emitted to notify views or other
        components of the change.

        Parameters
        ----------
        idx : int
            The index value to be removed from the DataFrame. This corresponds to the line_index in
            a MultiIndex.

        level : int, optional
            The level in the MultiIndex from which to remove the rows. Defaults to 1.
            This level should correspond to the position of the `idx` in the MultiIndex structure.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        - The `level` parameter determines which level of the MultiIndex is targeted for the row
            removal.
        - The `modelReset` signal is emitted to update any connected views or models about the
            changes in the data.

        Examples
        --------
        To delete rows where the line_index is 5 in a MultiIndex with level 1:
        >>> delete_rows(5, level=1)

        See Also
        --------
        pandas.DataFrame.drop : Method used to drop rows from the DataFrame.
        """
        self._dataframe = self._dataframe.drop(index=idx, level=level)
        self.modelReset.emit()

    def delete_rows_by_filenames(self, multi_index: list[str]) -> None:
        """
        Drop rows from the DataFrame based on a list of MultiIndex values.

        This method removes rows from the DataFrame where the index matches any of the provided
        MultiIndex values.
        After removing the rows, the `modelReset` signal is emitted to notify views or other
        components of the change.

        Parameters
        ----------
        multi_index : list of str
            A list of MultiIndex values (filenames) used to identify rows to be removed. Each entry
            in the list represents a filename that will be checked against the DataFrame's index.
            Only the existing MultiIndex values in the DataFrame will be dropped.

        Returns
        -------
        None
            This method does not return any value.

        Notes
        -----
        - The method drops rows where the MultiIndex matches any of the provided filenames.
        - The `modelReset` signal is emitted to update any connected views or models about the
            changes in the data.

        Examples
        --------
        To delete rows with filenames 'file1' and 'file2':
        >>> delete_rows_by_filenames(['file1', 'file2'])

        See Also
        --------
        pandas.DataFrame.drop : Method used to drop rows from the DataFrame.
        """
        df_index = self._dataframe.index
        existing_mi = []
        for i in multi_index:
            if i in df_index:
                existing_mi.append(i)
        self._dataframe = self._dataframe.drop(index=existing_mi, level=0)
        self.modelReset.emit()

    def delete_rows_multiindex(self, multi_index: tuple) -> None:
        """
        Drop table rows by MultiIndex

        Parameters
        ---------
        multi_index : MultiIndex
            filename, line_index, param_name
        """
        self._dataframe = self._dataframe.drop(index=multi_index, inplace=True, errors='ignore')
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

    def get_lines_indexes_by_filename(self, filename: str) -> list[int] | None:
        """
        Retrieve line indexes associated with a given filename from the DataFrame.

        This method returns a list of unique line indexes for the specified filename.
        If the DataFrame is empty, or if the filename does not exist in the DataFrame's index,
        or if no line indexes are found, the method returns `None`.

        Parameters
        ----------
        filename : str
            The filename used to filter the DataFrame and retrieve line indexes.

        Returns
        -------
        list of int | None
            A list of unique line indexes associated with the given filename, or `None` if:
            - The DataFrame is empty.
            - The filename is not present in the DataFrame's index.
            - No line indexes are found for the filename.

        Notes
        -----
        - The method filters the DataFrame based on the provided filename and then extracts unique
            line indexes.
        - If the filename is present but no line indexes are found, the method returns `None`.

        See Also
        --------
        pandas.DataFrame.loc : Method used to filter the DataFrame based on the filename.
        """
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

    def row_number_for_filtering(self, m_i: tuple[str, int]):
        """
        Retrieve the range of row numbers for a specific MultiIndex tuple in the DataFrame.

        This method returns a range object representing the row numbers in the DataFrame
        that correspond to the provided MultiIndex tuple. If the MultiIndex tuple is not found
        in the DataFrame's index or if the tuple's first element is not present, the method
        returns `None`.

        Parameters
        ----------
        m_i : tuple of (str, int)
            A tuple representing the MultiIndex to filter rows. The tuple should contain:
            - A string indicating the filename or identifier.
            - An integer representing the line index.

        Returns
        -------
        range | None
            A range object representing the row numbers for the given MultiIndex tuple, or `None`
            if:
            - The MultiIndex tuple is not found in the DataFrame's index.
            - The first element of the tuple is not present in the DataFrame's index.

        Notes
        -----
        - The range starts from the row number corresponding to the tuple `(m_i[0], m_i[1], 'a')`
          and extends for the number of rows equal to the count of rows associated with the given
            tuple.
        - The range will cover the row numbers starting from the first occurrence of the tuple
          `(m_i[0], m_i[1], 'a')` up to the end of the rows associated with the specified
            MultiIndex.

        Examples
        --------
        To get the range of row numbers for a MultiIndex `('example_file', 1)`:
        >>> range_of_rows = row_number_for_filtering(('example_file', 1))

        See Also
        --------
        pandas.DataFrame.index : The index of the DataFrame used to check for the MultiIndex
        presence.
        """
        df_index = self._dataframe.index
        if m_i[0] not in df_index or m_i not in df_index:
            return None
        row_count = len(self._dataframe.loc[m_i])
        list_idx = list(self._dataframe.index)
        row_num = list_idx.index((m_i[0], m_i[1], 'a'))
        return range(row_num, row_num + row_count)

    def lines_idx_by_x0_sorted(self) -> list[int]:
        """
        Returns indexes of lines sorted by x0 value cm-1
        """
        ser = self._dataframe.query('filename == "" and param_name == "x0"')['Value']
        vals = ser.sort_values().index.values
        return [i[1] for i in vals]


class PandasModelSmoothedDataset(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for smoothed dataset.
    """

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """
        Override method from QAbstractItemModel to return item flags for the given index.

        This method defines the item flags for each cell in the model, which determine
        the item's behavior and interaction within a view.

        Parameters
        ----------
        index : QModelIndex
            The index of the item for which flags are requested.

        Returns
        -------
        Qt.ItemFlags
            The flags that specify the item's properties and behaviors.
            - For items in the first column (index.column() == 0), the flags include
              `Qt.ItemIsSelectable`, `Qt.ItemIsEnabled`, and `Qt.ItemIsEditable`,
              allowing these items to be selected, edited, and enabled.
            - For items in other columns, the flags include `Qt.ItemIsSelectable`
              and `Qt.ItemIsEnabled`, allowing these items to be selected and enabled.

        Notes
        -----
        - `Qt.ItemIsSelectable` allows the item to be selected.
        - `Qt.ItemIsEnabled` allows the item to be enabled.
        - `Qt.ItemIsEditable` allows the item to be edited, which is only applicable
          to items in the first column.

        See Also
        --------
        QAbstractItemModel.flags : Base method from QAbstractItemModel which this method overrides.
        """
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Override method from QAbstractTableModel to return data for a specific cell from the pandas
        DataFrame.

        This method provides the data to be displayed or edited in a view for the cell at the given
        index.

        Parameters
        ----------
        index : QModelIndex
            The index of the item whose data is requested.
        role : Qt.ItemDataRole, optional
            The role for which data is requested. Defaults to Qt.DisplayRole.
            The role determines the type of data to be retrieved or displayed.

        Returns
        -------
        None | str | float
            The data for the cell at the specified index and role:
            - `Qt.DisplayRole`: Returns the data as a string for display purposes.
            - `Qt.EditRole`: Returns the data as a float for editing purposes. A `QDoubleSpinBox`
                is used to format the float with 5 decimal places.
            - `None`: For other roles, or if the index is invalid.

        Notes
        -----
        - If the `index` is not valid, `None` is returned.
        - For `Qt.DisplayRole`, the value is converted to a string representation.
        - For `Qt.EditRole`, the value is formatted as a float, which can be used in widgets such
            as `QDoubleSpinBox`.

        See Also
        --------
        QAbstractItemModel.data : Base method from QAbstractItemModel which this method overrides.
        QDoubleSpinBox : Widget used to format float values for editing.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.EditRole:
            double_spinbox = QDoubleSpinBox()
            double_spinbox.setValue(float(value))
            double_spinbox.setDecimals(5)
            return float(value)
        return None

    def setData(self, index, value, role):
        """
        Override method from QAbstractTableModel to set data for a specific cell in the pandas
        DataFrame.

        This method is used to set the value of a cell in the model when the user edits the data in
        the view.

        Parameters
        ----------
        index : QModelIndex
            The index of the item whose data is being set.
        value : float
            The new value to be set for the specified cell.
        role : Qt.ItemDataRole
            The role for which the data is being set. Only `Qt.EditRole` is handled in this method.

        Returns
        -------
        bool
            `True` if the data was successfully set, `False` otherwise.

        Notes
        -----
        - This method handles only `Qt.EditRole`. For other roles, `False` is returned.
        - The `value` is converted to a float before being set in the DataFrame.
        - After setting the data, the `dataChanged` signal is emitted to notify that the data has
            been updated.

        See Also
        --------
        QAbstractItemModel.setData : Base method from QAbstractItemModel which this method overrides
        """
        if role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True
        return False

    @property
    def filenames(self) -> list[str]:
        """
        Retrieve a list of filenames from the DataFrame.

        This property extracts the 'Filename' column from the DataFrame, filters out any empty
        values, and returns a list of non-empty filenames.

        Returns
        -------
        list of str
            A list containing the filenames from the 'Filename' column of the DataFrame. Only
            non-empty filenames are included.

        Notes
        -----
        - The 'Filename' column must be present in the DataFrame for this property to work correctly
        - If the 'Filename' column contains any empty or None values, they will be filtered out from
            the returned list.

        See Also
        --------
        DataFrame : The DataFrame from which the 'Filename' column is extracted.
        """
        filenames = self._dataframe['Filename']
        filenames = [i for i in filenames if i]
        return filenames


class PandasModelBaselinedDataset(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for baseline corrected dataset.
    """

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """
        Determine the item flags for the given QModelIndex.

        This method overrides the default behavior from QAbstractItemModel to specify the item flags
        for cells in the model. It sets different flags based on the column of the QModelIndex.

        Parameters
        ----------
        index : QModelIndex
            The index for which the item flags are requested. It specifies the row and column of the
            item within the model.

        Returns
        -------
        Qt.ItemFlags
            The item flags indicating the item properties for the given index. The flags determine
            whether the item is selectable, editable, or enabled.

        Notes
        -----
        - For column 0, the item is selectable, enabled, and editable.
        - For all other columns, the item is selectable and enabled but not editable.

        See Also
        --------
        QModelIndex : The index object used to access items within the model.
        Qt.ItemFlags : Flags that define the properties of items in the model.
        """
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Retrieve data for a specific item in the model, depending on the role.

        This method overrides the default behavior from `QAbstractTableModel` to return data from
        the internal pandas DataFrame based on the provided `QModelIndex` and role.

        Parameters
        ----------
        index : QModelIndex
            The index of the item in the model whose data is to be retrieved. The index specifies
            the row and column of the item.

        role : Qt.ItemDataRole, optional
            The role that specifies the type of data to be retrieved. The default is
                `Qt.DisplayRole`.
            Common roles include:
            - `Qt.DisplayRole`: Used to display the data in the view.
            - `Qt.EditRole`: Used for data that is editable.

        Returns
        -------
        str | float | None
            The data for the specified item and role. If the index is invalid or the role is not
            handled, the method returns `None`. For `Qt.DisplayRole`, it returns the data as a
            string. For `Qt.EditRole`, it returns the data as a float. Other roles return `None`.

        Notes
        -----
        - The `Qt.DisplayRole` returns data as a string for display purposes. If the data is a
            `dict`,
          it formats the `dict` into a string describing the style and width.
        - The `Qt.EditRole` returns the data as a float for editing purposes. A `QDoubleSpinBox` is
          used to set and display the float value with up to 5 decimal places.

        See Also
        --------
        QModelIndex : The index object used to identify the item in the model.
        Qt.ItemDataRole : Enum specifying the role of the data to be retrieved.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.EditRole:
            double_spinbox = QDoubleSpinBox()
            double_spinbox.setValue(float(value))
            double_spinbox.setDecimals(5)
            return float(value)
        return None

    def setData(self, index, value, role):
        """
        Set the data for a specific item in the model based on the provided role.

        This method overrides the default behavior from `QAbstractTableModel` to update data in the
        internal pandas DataFrame. It emits the `dataChanged` signal if the data was successfully
        updated.

        Parameters
        ----------
        index : QModelIndex
            The index of the item in the model where data is to be set. The index specifies the row
            and column of the item.

        value : float
            The new value to be set in the model. The value is converted to a float before updating.

        role : Qt.ItemDataRole
            The role that specifies the type of data being set. The default is `Qt.EditRole`.

        Returns
        -------
        bool
            `True` if the data was successfully set; `False` otherwise.

        Notes
        -----
        - This method only handles `Qt.EditRole`. If the role is not `Qt.EditRole`, the method
            returns `False`.
        - The `Qt.EditRole` is used for setting editable data. In this implementation, the value is
          converted to a float before updating the DataFrame.
        - After updating the data, the `dataChanged` signal is emitted to notify any views that the
            data
          has changed.

        See Also
        --------
        QModelIndex : The index object used to identify the item in the model.
        Qt.ItemDataRole : Enum specifying the role of the data to be set.
        """
        if role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True
        return False

    @property
    def filenames(self) -> list[str]:
        """
        Get the list of filenames from the DataFrame.

        This property retrieves the filenames stored in the 'Filename' column of the internal pandas
        DataFrame.
        It filters out any empty or null values, returning a list of non-empty filenames.

        Returns
        -------
        list[str]
            A list of filenames from the DataFrame, with empty or null values excluded.

        Notes
        -----
        - The property assumes that the DataFrame contains a column named 'Filename'.
        - If the DataFrame is empty or the 'Filename' column does not exist, this will return an
            empty list.

        See Also
        --------
        pandas.DataFrame : The DataFrame from which filenames are extracted.
        """
        filenames = self._dataframe['Filename']
        filenames = [i for i in filenames if i]
        return filenames


class PandasModelDeconvolutedDataset(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for decomposed dataset.
    """

    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """
        Override method from QAbstractItemModel to return the item flags for a given index.

        Determines the flags for an item based on its index and column, specifying its interaction
        capabilities.

        Parameters
        ----------
        index : QModelIndex
            The index for which to determine the flags.

        Returns
        -------
        Qt.ItemFlags
            The flags for the item at the given index. The flags specify how the item can be
            interacted with.

        Notes
        -----
        - If the column is 0, the item is selectable, enabled, and editable.
        - For other columns, the item is only selectable and enabled.

        See Also
        --------
        QAbstractItemModel : The base class from which this method is derived.
        QModelIndex : Represents the model index for which the flags are being determined.
        Qt.ItemFlags : The type used to specify item interaction flags.
        """
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Override method from QAbstractTableModel to return data from the pandas DataFrame based on
        the role.

        Retrieves the data for a given cell from the DataFrame and formats it according to the
        specified role.

        Parameters
        ----------
        index : QModelIndex
            The index of the cell for which data is requested.
        role : Qt.ItemDataRole, optional
            The role indicating how the data should be formatted. The default is Qt.DisplayRole.

        Returns
        -------
        None | str | float
            The data for the cell as a string for display roles or as a float for edit roles.
            Returns None for other roles.

        Notes
        -----
        - If the role is `Qt.DisplayRole`, the data is returned as a string representation of the
            value.
        - If the role is `Qt.EditRole`, the data is returned as a float, formatted with 5 decimal
            places.
        - For other roles, the method returns None.

        See Also
        --------
        QAbstractTableModel : The base class from which this method is derived.
        QModelIndex : Represents the model index for which data is being retrieved.
        Qt.ItemDataRole : Enum indicating the role of the data being retrieved.
        QDoubleSpinBox : Widget used to set float values in the edit role.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.EditRole:
            double_spinbox = QDoubleSpinBox()
            double_spinbox.setValue(float(value))
            double_spinbox.setDecimals(5)
            return float(value)
        return None

    def setData(self, index: QModelIndex, value, role: Qt.ItemDataRole):
        """
        Override method from QAbstractTableModel to set data in the pandas DataFrame.

        Updates the data in the DataFrame for a specific cell based on the provided role and emits a
        signal to indicate the change.

        Parameters
        ----------
        index : QModelIndex
            The index of the cell where data should be set.
        value : float
            The new value to be set in the cell.
        role : Qt.ItemDataRole
            The role indicating the type of data being set. Only `Qt.EditRole` is handled by this
            method.

        Returns
        -------
        bool
            Returns `True` if the data was successfully set; `False` otherwise.

        Notes
        -----
        - This method only handles `Qt.EditRole`. Data for other roles is not modified.
        - After updating the DataFrame, the `dataChanged` signal is emitted to notify views of the
            change.

        See Also
        --------
        QAbstractTableModel : The base class from which this method is derived.
        QModelIndex : Represents the model index for which data is being set.
        Qt.ItemDataRole : Enum indicating the role of the data being set.
        """
        if role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True
        return False

    @property
    def filenames(self) -> list[str]:
        """
        Get a list of filenames from the DataFrame.

        Returns
        -------
        list[str]
            A list of filenames extracted from the 'Filename' column of the DataFrame.
            Filenames are filtered to exclude any empty or None values.

        Notes
        -----
        - The filenames are retrieved from the 'Filename' column of the DataFrame.
        - This property assumes that the DataFrame contains a column named 'Filename'.

        See Also
        --------
        pandas.DataFrame : The DataFrame from which filenames are extracted.
        """
        filenames = self._dataframe['Filename']
        filenames = [i for i in filenames if i]
        return filenames

    @property
    def features(self) -> list[str]:
        """
        Get a list of feature names from the DataFrame columns.

        Returns
        -------
        list[str]
            A list of feature names extracted from the columns of the DataFrame,
            excluding the first two columns.

        Notes
        -----
        - The feature names are derived from the columns of the DataFrame starting from the third
            column.
        - This property assumes that the DataFrame has at least three columns.


        See Also
        --------
        pandas.DataFrame : The DataFrame from which feature names are extracted.
        """
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


class PandasModelIgnoreDataset(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for ignored features dataset.
    """

    def __init__(self, parent, dataframe: DataFrame, checked: dict):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe
        self._checked = checked

    def flags(self, _: QModelIndex) -> Qt.ItemFlags:
        """
        Override method from QAbstractItemModel to specify item flags.

        Determines the flags for the item represented by the provided index.

        Parameters
        ----------
        _ : QModelIndex
            The index for which the flags are to be returned. This parameter is not used in the
            method.

        Returns
        -------
        Qt.ItemFlags
            Flags indicating the item is selectable, enabled, and checkable by the user.

        Notes
        -----
        - `Qt.ItemIsSelectable` allows the item to be selectable.
        - `Qt.ItemIsEnabled` ensures that the item is enabled and interactive.
        - `Qt.ItemIsUserCheckable` makes the item checkable by the user.

        See Also
        --------
        QAbstractItemModel : The base class for model implementations in Qt.
        Qt.ItemFlags : Flags used to define item behavior in Qt's model-view framework.
        """
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Override method from QAbstractTableModel to return data for a specific cell.

        This method retrieves data from the pandas DataFrame based on the given index and role.

        Parameters
        ----------
        index : QModelIndex
            The index of the cell for which data is requested. Must be valid for the method to
            return a value.
        role : Qt.ItemDataRole, optional
            The role of the data to retrieve. Default is `Qt.DisplayRole`.

        Returns
        -------
        None | str | Qt.CheckState
            The data associated with the given index and role:
            - For `Qt.DisplayRole` and `Qt.ItemDataRole.EditRole`, returns the data as a string.
            - For `Qt.CheckStateRole`, returns `Qt.Checked` or `Qt.Unchecked` based on whether the
                item is checked.
            - `None` is returned if the index is invalid or the role is not handled by the method.

        Notes
        -----
        - `Qt.DisplayRole` and `Qt.ItemDataRole.EditRole` return the cell's value as a string.
        - `Qt.CheckStateRole` returns a check state depending on whether the item's feature is
            checked or not.
        - If `index` is invalid, `None` is returned.

        See Also
        --------
        QAbstractTableModel : Base class for models in Qt's Model/View framework.
        Qt.ItemDataRole : Enumeration of different data roles used by model/view classes.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.EditRole:
            return str(value)
        if role == Qt.CheckStateRole:
            checked_name = self._dataframe.iloc[index.row()].Feature
            return Qt.Checked if checked_name not in self._checked or self._checked[
                checked_name] else Qt.Unchecked
        return None

    def setData(self, index, value, role):
        """
        Override method from QAbstractTableModel to set data for a specific cell.

        This method updates the data in the pandas DataFrame based on the given index, value, and
        role.

        Parameters
        ----------
        index : QModelIndex
            The index of the cell to update. The index must be valid for the data to be set.
        value : Any
            The new value to set for the cell. The type of `value` depends on the role.
        role : Qt.ItemDataRole
            The role of the data being set. Determines how the `value` should be applied.

        Returns
        -------
        bool
            `True` if the data was successfully set; `False` otherwise.

        Notes
        -----
        - This method currently supports updating check states via `Qt.CheckStateRole`.
        - When `role` is `Qt.CheckStateRole`, the method updates the check state based on the
            provided `value`.
          - If `value` is `2`, the item is marked as checked.
          - If `value` is not `2`, the item is marked as unchecked.
        - For other roles or invalid indices, the method returns `False`.

        See Also
        --------
        QAbstractTableModel : Base class for models in Qt's Model/View framework.
        Qt.ItemDataRole : Enumeration of different data roles used by model/view classes.
        """
        if role == Qt.CheckStateRole:
            checked = value == 2
            checked_name = self._dataframe.iloc[index.row()].Feature
            self._checked[checked_name] = checked
            return True
        return False

    @property
    def checked(self) -> dict:
        """
        Retrieve the current check states for items in the model.

        This property returns a dictionary mapping item names (or indices) to their check states.

        Returns
        -------
        dict
            A dictionary where keys are item names or indices, and values are booleans indicating
            whether the items are checked (`True`) or unchecked (`False`).

        Notes
        -----
        - The dictionary reflects the current check states of items managed by the model.
        - The structure of the dictionary is dependent on how items are identified and
            checked/unchecked in the model. For example, keys might be item names or unique indices,
            depending on the model's implementation.

        See Also
        --------
        setData : Method to update the check states of items.
        """
        return self._checked

    def set_checked(self, checked: dict) -> None:
        """
        Update the check states of items in the model.

        This method sets the check states based on the provided dictionary. The dictionary should
        map item names or indices to their desired check states.

        Parameters
        ----------
        checked : dict
            A dictionary where keys are item names or indices, and values are booleans indicating
            whether the items should be checked (`True`) or unchecked (`False`).

        Notes
        -----
        - This method directly updates the internal check states of the model.
        - It is assumed that the keys in the `checked` dictionary correspond to items managed by the
            model.


        See Also
        --------
        checked : Property to retrieve the current check states of items.
        """
        self._checked = checked

    def set_all_features_checked(self) -> None:
        """
        self._checked has features which was changed and has structure like
        {'k1001.00_a': False,  ....}
        This function sets all values to True

        Returns
        -------
        None
        """
        for k in self._checked.keys():
            self._checked[k] = True

    @property
    def ignored_features(self) -> list[str]:
        """
        Retrieve a list of features that are currently ignored.

        This property returns a list of feature names that are marked as unchecked (ignored) in the
        model. The ignored features are identified based on the `_checked` dictionary where the
        value is `False`.

        Returns
        -------
        list of str
            A list of feature names that are ignored. Only features with a `False` value in the
            `_checked` dictionary are included in this list.

        Notes
        -----
        - The `_checked` dictionary should have feature names as keys and boolean check states as
            values.
        - This property provides a way to quickly access which features are not currently selected.

        Examples
        --------
        If the `_checked` dictionary is `{'feature1': True, 'feature2': False, 'feature3': True}`,
        then `ignored_features` will return `['feature2']`.

        See Also
        --------
        checked : Property that provides the current check states of features.
        """
        return [k for k, v in self._checked.items() if not v]

    def features_by_order(self) -> list[str]:
        """
        Returns sorted by Score activated features.
        """
        df_sort = self._dataframe.sort_values(by='Score', ascending=False)
        active_features = self._active_features
        return df_sort['Feature'][df_sort['Feature'].isin(active_features)] \
            if active_features else df_sort['Feature']

    @property
    def n_features(self) -> int:
        """
        Number of active features in the model.

        This property returns the count of features that are considered active. If there are active
        features specified in the `_active_features` list, it returns the length of this list.
        Otherwise, it defaults to returning the total number of rows in the `_dataframe`.

        Returns
        -------
        int
            The number of active features. If `_active_features` is not empty, returns its length.
            If `_active_features` is empty or `None`, returns the total number of rows in
            `_dataframe`.

        Notes
        -----
        - The property relies on `_active_features` to determine the count of features. If
            `_active_features` is `None` or empty, it falls back to using the number of rows in
            `_dataframe`.
        - The `_dataframe` should be a DataFrame where the number of rows corresponds to the total
            number of features.

        Examples
        --------
        If `_active_features` is `['feature1', 'feature2', 'feature3']`, `n_features` will return
            `3`.

        If `_active_features` is `None` or empty and `_dataframe` has 10 rows, `n_features` will
            return `10`.

        See Also
        --------
        _active_features : A list of active features considered by the model.
        _dataframe : The DataFrame that holds all feature data.
        """
        active_features = self._active_features
        return len(active_features) if active_features else self._dataframe.shape[0]

    @property
    def _active_features(self) -> list:
        """
        List of active features based on the checked status.

        This property filters and returns a list of feature names from `_checked` where the value is
        `True`,
        indicating that these features are currently active.

        Returns
        -------
        list
            A list of feature names that are marked as active. These are the keys from `_checked`
            where the value is `True`.

        Notes
        -----
        - The `_checked` dictionary is expected to contain feature names as keys and their checked
            status as boolean values.
        - Only the features that are checked (i.e., where the value in `_checked` is `True`) are
            included in the result list.

        See Also
        --------
        _checked : Dictionary containing feature names as keys and their checked status as boolean
            values.
        """
        return [k for k, v in self._checked.items() if v]


class PandasModelDescribeDataset(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for describe dataset.
    """
    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, _: QModelIndex) -> Qt.ItemFlags:
        """
        Returns the flags for the given index in the model.

        This method provides the flags that indicate the item state and behavior
        for the specified index in the model. In this implementation, it returns
        `Qt.ItemIsSelectable`, which means the item can be selected.

        Parameters
        ----------
        _: QModelIndex
            The index for which to return the flags. The index is not used in this implementation.

        Returns
        -------
        Qt.ItemFlags
            A combination of flags that describe the item properties. In this case, it returns
            `Qt.ItemIsSelectable`, which allows the item to be selected by the user.

        Notes
        -----
        - The flags determine the item’s interactive features, such as whether it is selectable,
            editable, or checkable.
        - This implementation does not use the provided `index` parameter and only returns
            `Qt.ItemIsSelectable`.

        See Also
        --------
        Qt.ItemFlags : Enum that defines the different flags for item properties.
        """
        return Qt.ItemIsSelectable

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Retrieve the data for a specific item in the model.

        This method returns the data for the item at the specified index from the
        underlying pandas DataFrame. The data is formatted according to the role
        provided.

        Parameters
        ----------
        index : QModelIndex
            The index of the item for which to retrieve the data. It should be a
            valid index within the model. If the index is not valid, the method
            returns `None`.
        role : Qt.ItemDataRole, optional
            The role for which the data is requested. It determines how the data
            should be formatted. The default value is `Qt.DisplayRole`.

        Returns
        -------
        str or None
            The formatted data for the item at the given index. If the role is
            `Qt.DisplayRole`, the data is formatted as a string with the value
            rounded to five decimal places. For other roles, the method returns
            `None`.

        Notes
        -----
        - This method currently handles the `Qt.DisplayRole` role by converting the
          data to a string and rounding it to five decimal places.
        - Other roles are not handled in this implementation, so the method returns
          `None` for those cases.

        See Also
        --------
        Qt.ItemDataRole : Enum that defines the roles used to request different
                          types of data from the model.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(np.round(value, 5))
        return None

    def setData(self, index: QModelIndex, value, role: Qt.ItemDataRole) -> bool:
        """
        Set the data for a specific item in the model.

        This method updates the data for the item at the specified index in the
        underlying pandas DataFrame. It currently does not modify the data,
        and always returns `False`.

        Parameters
        ----------
        index : QModelIndex
            The index of the item to be updated. The index should be valid and
            correspond to an item in the model.
        value : any
            The new value to set for the item. The type of the value should match
            the data type expected by the model for the specified role.
        role : Qt.ItemDataRole
            The role for which the data is being set. It determines how the data
            should be interpreted and processed. The default value is
            `Qt.EditRole`.

        Returns
        -------
        bool
            `False` is always returned by this method. This indicates that the
            data was not updated. The method currently does not handle any data
            updates.

        Notes
        -----
        - This method is a placeholder that does not perform any actual data
          modification. It always returns `False`, signaling that no data change
          has occurred.
        - To enable data updates, this method should be implemented to handle
          specific roles and update the model's underlying data accordingly.

        Examples
        --------
        If this method is called with a valid index, value, and role, it will
        return `False` and not modify the data in the model.

        See Also
        --------
        Qt.ItemDataRole : Enum that defines the roles used to set different types
                          of data in the model.
        """
        return False


class PandasModelPredictTable(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for predict dataset.
    """
    def __init__(self, context, dataframe: DataFrame):
        super().__init__(dataframe)
        self.context = context
        self._dataframe = dataframe

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """
        Return the item flags for a given index.

        This method specifies the flags for each item in the model, which determine
        the item's behavior in terms of selection, editing, and enabling. The flags
        define what actions can be performed on the item and how it can be interacted with.

        Parameters
        ----------
        index : QModelIndex
            The index of the item for which the flags are being requested. The index
            should be valid and correspond to an item in the model.

        Returns
        -------
        Qt.ItemFlags
            The item flags for the specified index. This is a combination of flags
            from the `Qt.ItemFlag` enum that determine the item's properties and
            behavior.

        Notes
        -----
        - If the index is in the first column (index.column() == 0), the item is
          selectable, enabled, and editable.
        - For all other columns, the item is only selectable and enabled, but not
          editable.

        Examples
        --------
        If `index.column()` is `0`, the method will return a combination of flags that
        allow the item to be selectable, enabled, and editable. For any other column,
        it will return a combination of flags that allow the item to be selectable and
        enabled but not editable.

        See Also
        --------
        Qt.ItemFlag : Enum that defines flags used to describe the item's properties.
        """
        if index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Return the data for the given index and role from the pandas DataFrame.

        This method retrieves data from the DataFrame based on the specified index
        and role. It provides different data formats depending on the role requested
        (e.g., display, foreground color).

        Parameters
        ----------
        index : QModelIndex
            The index of the item in the model. It should be a valid index within the
            model's range.

        role : Qt.ItemDataRole, optional
            The role for which the data is requested. This determines the type of data
            to be returned. The default is `Qt.ItemDataRole.DisplayRole`.

        Returns
        -------
        any
            The data for the given index and role. The type of data returned depends
            on the role:
            - `Qt.DisplayRole`: The string representation of the value in the DataFrame.
            - `Qt.ItemDataRole.ForegroundRole`: The color associated with the data
              based on the group number, if applicable.

        Notes
        -----
        - If the role is `Qt.ItemDataRole.ForegroundRole` and the column is not the first
          column (index.column() != 0), the method retrieves the foreground color for
          the item based on the group number. This is done by extracting the group number
          from the string representation of the value and using the context to get the
          associated color.

        Examples
        --------
        If `index.column() == 0`, the method returns the string representation of the value
        in the DataFrame. For other columns, if the role is `Qt.ItemDataRole.ForegroundRole`,
        the method returns a color based on the group number extracted from the value.

        See Also
        --------
        Qt.ItemDataRole : Enum defining the roles used to request specific types of data.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        if role == Qt.ItemDataRole.ForegroundRole and index.column() != 0:
            class_str = str(value).split(' ', maxsplit=1)[0]
            return self.context.group_table.get_color_by_group_number(class_str)
        return None

    def setData(self, index: QModelIndex, value, role: Qt.ItemDataRole) -> bool:
        """
        Set the data for the given index and role in the pandas DataFrame.

        This method is used to set data in the model for a specific index and role.
        It updates the underlying DataFrame if the role is appropriate for editing.

        Parameters
        ----------
        index : QModelIndex
            The index of the item in the model that is being updated. It should be a
            valid index within the model's range.

        value : any
            The new value to be set for the item at the specified index. The type of
            value depends on the role.

        role : Qt.ItemDataRole
            The role that determines how the data is used or processed. Only certain
            roles are editable, such as `Qt.EditRole`.

        Returns
        -------
        bool
            Returns `True` if the data was successfully set; otherwise, `False`.

        Notes
        -----
        - This method currently returns `False` for all roles, indicating that data
          changes are not supported or not implemented.
        - For `Qt.EditRole`, if the data was to be updated, it would involve updating
          the DataFrame and emitting the `dataChanged` signal.

        Examples
        --------
        The method does not perform any updates in its current implementation and
        always returns `False`.

        See Also
        --------
        Qt.ItemDataRole : Enum defining the roles used to request or set specific types of data.
        """
        return False


class PandasModelPCA(PandasModel):
    """
    A model to interface a Qt view with pandas dataframe for PCA dataset.
    """
    def __init__(self, parent, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def flags(self, _: QModelIndex) -> Qt.ItemFlags:
        """
        Return the item flags for a given index.

        This method provides the flags that determine the item's behavior in the model,
        such as whether it can be selected, edited, or interacted with in other ways.

        Parameters
        ----------
        _ : QModelIndex
            The index of the item for which the flags are being queried. This parameter
            is currently unused in this implementation.

        Returns
        -------
        Qt.ItemFlags
            The flags that specify the item's interaction capabilities.

        Notes
        -----
        - The flags returned are:
          - `Qt.ItemIsSelectable`: Indicates that the item can be selected.
          - `Qt.ItemIsEnabled`: Indicates that the item is enabled and can be interacted with.
        - Editing and checking flags are not enabled in this implementation.

        Examples
        --------
        >>> model.flags(QModelIndex())
        <Qt.ItemFlags: 3>

        See Also
        --------
        Qt.ItemFlags : Enum defining various item flags.
        """
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """
        Retrieve the data for a given index and role from the pandas DataFrame.

        This method is used to return data to be displayed or edited in a model view,
        based on the role requested.

        Parameters
        ----------
        index : QModelIndex
            The index of the item for which data is being retrieved. This includes
            the row and column positions in the DataFrame.
        role : Qt.ItemDataRole, optional
            The role of the data being requested. This determines how the data is
            formatted or presented. The default is `Qt.DisplayRole`.

        Returns
        -------
        Union[str, None]
            The data corresponding to the requested role. If the index is not valid,
            or the role is not handled, `None` is returned. For `Qt.DisplayRole`,
            the data is returned as a string.

        Notes
        -----
        - `Qt.DisplayRole`: This role is used to display data. The data is converted to a
          string format for display purposes.
        - Other roles are not handled in this implementation; thus, `None` is returned
          for those roles.

        Examples
        --------
        >>> model.data(QModelIndex(), Qt.DisplayRole)
        'example data'

        See Also
        --------
        Qt.ItemDataRole : Enum defining various data roles.
        """
        if not index.isValid():
            return None
        value = self._dataframe.iloc[index.row(), index.column()]

        if role == Qt.DisplayRole:
            return str(value)
        return None


class DoubleSpinBoxDelegate(QStyledItemDelegate):
    """
    A delegate for editing floating-point values in a `QTableView` using a `QDoubleSpinBox`.

    This delegate provides a custom editor for floating-point values and emits signals
    when values are changed. It also integrates with an undo stack to allow undo/redo
    functionality for changes made to the data.

    Attributes
    ----------
    sigLineParamChanged : Signal
        Emitted when a line parameter is changed. The signal carries the new float value,
        the line index, and the parameter name.

    Methods
    -------
    __init__(rs)
        Initializes the delegate with the given parameters.
    createEditor(parent, option, index)
        Creates and returns a `QDoubleSpinBox` for editing the cell data.
    setModelData(editor, model, index)
        Updates the model with the data from the editor and handles any post-edit actions.
    editing_finished()
        Emits the `commitData` signal to finalize editing.
    """
    sigLineParamChanged = Signal(float, int, str)

    def __init__(self, rs):
        """
        Initializes the delegate with the given parameter.

        Parameters
        ----------
        rs : object
            An object that provides access to relevant resources (e.g., the main window or model).
        """
        super().__init__()
        self.mw = rs

    def createEditor(self, parent, _: QStyleOptionViewItem, index: QModelIndex):
        """
        Creates and configures a `QDoubleSpinBox` editor for the cell at the specified index.

        The range and limits of the spin box are set based on the parameter name associated
        with the cell.

        Parameters
        ----------
        parent : QWidget
            The parent widget for the editor.
        _ : QStyleOptionViewItem
            Style options for the item.
        index : QModelIndex
            The index of the item to be edited.

        Returns
        -------
        QDoubleSpinBox
            The configured `QDoubleSpinBox` editor.
        """
        param_name = self.mw.ui.fit_params_table.model().row_data(index.row()).name[2]
        min_limit = -1e6
        max_limit = 1e6
        if param_name == 'l_ratio':
            min_limit = 0.0
            max_limit = 1.0
        elif param_name in ['expon', 'beta']:
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

    def setModelData(self, editor: QDoubleSpinBox, model: QAbstractItemModel, index: QModelIndex) \
            -> None:
        """
        Updates the model with the value from the editor and emits a signal if necessary.

        The method checks if the new value differs from the current value. If it does, it updates
        the model and emits a signal to indicate that a line parameter has changed. It also
        creates an undo command for the change.

        Parameters
        ----------
        editor : QDoubleSpinBox
            The editor containing the new value.
        model : QAbstractItemModel
            The model that needs to be updated.
        index : QModelIndex
            The index of the item being edited.
        """
        new_float = editor.value()
        current_float = model.cell_data_by_index(index)
        if new_float == current_float:
            return
        model.setData(index, new_float, Qt.EditRole)
        if index.column() != 1:
            return
        _, line_index, param_name = model.dataframe.iloc[index.row()].name
        self.sigLineParamChanged.emit(new_float, line_index, param_name)
        command = CommandDeconvLineParameterChanged((index, new_float, current_float),
                                                    self.mw.context, text=f"Edit line {index}",
                                                    **{'stage': self, 'model': model,
                                                       'line_index': line_index,
                                                       'param_name': param_name})
        self.mw.context.undo_stack.push(command)

    @pyqtSlot()
    def editing_finished(self):
        """
        Emits the `commitData` signal to finalize the editing of the current editor.
        """
        self.commitData.emit(self.sender())


class IntervalsTableDelegate(TableItemDelegate):
    """
    A delegate for editing floating-point values in a table view using a `QDoubleSpinBox`.

    This delegate is specifically designed for editing interval values in a table. It provides
    a custom editor with a `QDoubleSpinBox`, handles updates to the model, and integrates with
    an undo stack for command management.

    Attributes
    ----------
    context : object
        An object providing access to relevant resources (e.g., undo stack and other
        context-specific data).

    Methods
    -------
    __init__(parent, context)
        Initializes the delegate with the given parent and context.
    createEditor(parent, option, index)
        Creates and configures a `QDoubleSpinBox` editor for editing the cell data.
    setModelData(editor, model, index)
        Updates the model with the data from the editor and handles any post-edit actions.
    editing_finished()
        Emits the `commitData` signal to finalize editing.
    """
    def __init__(self, parent, context):
        """
        Initializes the delegate with the given parent and context.

        Parameters
        ----------
        parent : QWidget
            The parent widget for the delegate.
        context : object
            An object that provides access to the undo stack and other context-specific data.
        """
        super().__init__(parent)
        self.context = context

    def createEditor(self, parent, option: QStyleOptionViewItem, index: QModelIndex) \
            -> QDoubleSpinBox:
        """
        Creates and configures a `QDoubleSpinBox` editor for the cell at the specified index.

        The spin box is configured to allow values between 0 and 10,000 with up to 5 decimal places.

        Parameters
        ----------
        parent : QWidget
            The parent widget for the editor.
        option : QStyleOptionViewItem
            Style options for the item.
        index : QModelIndex
            The index of the item to be edited.

        Returns
        -------
        QDoubleSpinBox
            The configured `QDoubleSpinBox` editor.
        """
        editor = QDoubleSpinBox(parent)
        editor.setDecimals(5)
        editor.setSingleStep(1.)
        editor.setMinimum(0.)
        editor.setMaximum(10_000.)
        editor.editingFinished.connect(self.editing_finished)
        return editor

    def setModelData(self, editor: QDoubleSpinBox, model: QAbstractItemModel, index: QModelIndex) \
            -> None:
        """
        Updates the model with the value from the editor and creates an undo command if necessary.

        The method checks if the new value differs from the current value. If it does, it updates
        the model and creates an undo command for the change. This command is pushed to the undo
        stack.

        Parameters
        ----------
        editor : QDoubleSpinBox
            The editor containing the new value.
        model : QAbstractItemModel
            The model that needs to be updated.
        index : QModelIndex
            The index of the item being edited.
        """
        new_float = editor.value()
        current_float = model.cell_data_by_index(index)
        if new_float == current_float:
            return
        if index.column() != 0:
            return
        command = CommandFitIntervalChanged(new_float, self.context,
                                            f"Edit interval {index}",
                                            **{'index': index, 'model': model})
        self.context.undo_stack.push(command)

    @pyqtSlot()
    def editing_finished(self):
        """
        Emits the `commitData` signal to finalize the editing of the current editor.
        """
        self.commitData.emit(self.sender())


class ComboDelegate(QStyledItemDelegate):
    """
    A delegate for editing cell values with a `QListWidget` as the editor.

    This delegate provides a custom editor in the form of a `QListWidget` for selecting from
    a list of line types. It handles updates to the model, emits signals when the selection
    changes, and manages editor dimensions and positioning.

    Attributes
    ----------
    sigLineTypeChanged : Signal
        emitted when the selected line type changes.
    editorItems : set[str]
        A set of line types to be displayed in the editor.
    height : int
        The height of the editor in pixels.
    width : int
        The width of the editor in pixels.
    _cursor_pos : QPoint
        The position of the cursor when the editor is created.
    _cursor : QCursor
        The current cursor object.

    Methods
    -------
    __init__(line_types)
        Initializes the delegate with the provided line types.
    createEditor(parent, option, index)
        Creates a `QListWidget` editor for selecting line types.
    setEditorData(editor, index)
        Sets up the editor with the data from the model.
    setModelData(editor, model, index)
        Updates the model with the data from the editor and emits the signal if the data changes.
    current_item_changed()
        Emits the `commitData` signal when the current item in the editor changes.
    """
    sigLineTypeChanged = Signal(str, str, int)

    def __init__(self, line_types: set[str]):
        """
        Initializes the delegate with the provided line types.

        Parameters
        ----------
        line_types : set[str]
            A set of line types to be displayed in the editor.
        """
        super().__init__()
        self.editorItems = line_types
        self.height = 30
        self.width = 140
        self._cursor_pos = QPoint(0, 0)
        self._cursor = QCursor()

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex):
        """
        Creates a `QListWidget` editor for selecting line types.

        The editor is positioned based on the current cursor position.

        Parameters
        ----------
        parent : QWidget
            The parent widget for the editor.
        option : QStyleOptionViewItem
            Style options for the item.
        index : QModelIndex
            The index of the item being edited.

        Returns
        -------
        QListWidget
            The configured `QListWidget` editor.
        """
        editor = QListWidget(parent)
        self._cursor_pos = parent.mapFromGlobal(self._cursor.pos())
        editor.currentItemChanged.connect(self.current_item_changed)
        return editor

    def setEditorData(self, editor: QListWidget, index: QModelIndex) -> None:
        """
        Sets up the editor with the data from the model.

        The editor is populated with items from `editorItems`. The currently selected item
        in the editor is set to match the value of the cell.

        Parameters
        ----------
        editor : QListWidget
            The editor widget to be set up.
        index : QModelIndex
            The index of the item being edited.
        """
        z = 0
        for item in self.editorItems:
            ai = QListWidgetItem(item)
            editor.addItem(ai)
            if item == index.data():
                editor.setCurrentItem(editor.item(z))
            z += 1
        editor.setGeometry(self._cursor_pos.x(), self._cursor_pos.y(), self.width,
                           self.height * len(self.editorItems))

    def setModelData(self, editor: QListWidget, model: QAbstractItemModel, index: QModelIndex) \
            -> None:
        """
        Updates the model with the data from the editor and emits the signal if the data changes.

        Parameters
        ----------
        editor : QListWidget
            The editor widget containing the new value.
        model : QAbstractItemModel
            The model to be updated.
        index : QModelIndex
            The index of the item being edited.
        """
        new_text = editor.currentItem().text()
        current_text = model.cell_data_by_index(index)
        if current_text == new_text:
            return
        model.setData(index, new_text, Qt.EditRole)
        self.sigLineTypeChanged.emit(new_text, current_text, index.row())
        editor.close()

    @pyqtSlot()
    def current_item_changed(self):
        """
        Emits the `commitData` signal when the current item in the editor changes.
        """
        self.commitData.emit(self.sender())
