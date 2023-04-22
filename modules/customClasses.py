from pathlib import Path
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, concat, ExcelWriter
from pyqtgraph import arrayToQPath, mkPen
from qtpy.QtCore import Qt, QRectF, QAbstractTableModel, QModelIndex, Signal, Slot as pyqtSlot, QPoint
from qtpy.QtGui import QColor, QPainterPath, QCursor, QCloseEvent
from qtpy.QtWidgets import QGraphicsPathItem, QGraphicsItem, QDoubleSpinBox, QStyledItemDelegate, QListWidget, \
    QListWidgetItem, QWidget, QMainWindow, QDialog, QComboBox, QPushButton, QFrame, QFormLayout

from modules import curve_properties_ui
from modules.settingsui import Ui_Form
from modules.spec_functions import random_rgb
from modules.undo_classes import CommandUpdateTableCell, CommandDeconvLineParameterChanged, CommandFitIntervalChanged


def _fourierTransform(x_in: np.ndarray, y_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Perform Fourier transform. If x values are not sampled uniformly,
    # then use np.interp to resample before taking fft.
    x_out = []
    y_out = []
    for i in range(len(x_in)):
        x = x_in[i]
        y = y_in[i]
        dx = np.diff(x)
        uniform = not np.any(np.abs(dx - dx[0]) > (abs(dx[0]) / 1000.))
        if not uniform:
            x2 = np.linspace(x[0], x[-1], len(x))
            y = np.interp(x2, x, y)
            x = x2
        n = y.size
        f = np.fft.rfft(y) / n
        d = float(x[-1] - x[0]) / (len(x) - 1)
        x = np.fft.rfftfreq(n, d)
        x_out.append(x)
        y = np.abs(f)
        y_out.append(y)
    return np.array(x_out), np.array(y_out)


def apply_log_mapping(x: np.ndarray, y: np.ndarray, log_mode: tuple[bool, bool]) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies a logarithmic mapping transformation (base 10) if requested for the respective axis.
    This replaces the internal data. Values of ``-inf`` resulting from zeros in the original dataset are
    replaced by ``np.NaN``.

    Parameters
    ----------
    x: 2D array with all x values
    y: 2D array with all y values
    log_mode: tuple or list of two bool
        A `True` value requests log-scale mapping for the x and y-axis (in this order).
    """
    if log_mode[0]:
        with catch_warnings():
            simplefilter("ignore", RuntimeWarning)
            x = np.log10(x)
        nonfinites = ~np.isfinite(x)
        if nonfinites.any():
            x[nonfinites] = np.nan  # set all non-finite values to NaN
            idx_list = []
            for i in x:
                idx_list.append(np.nanargmin(i))
            idx = max(idx_list)
            x = x.T[idx:].T
            y = y.T[idx:].T
    if log_mode[1]:
        with catch_warnings():
            simplefilter("ignore", RuntimeWarning)
            y = np.log10(y)
        nonfinites = ~np.isfinite(y)
        if nonfinites.any():
            y[nonfinites] = np.nan  # set all non-finite values to NaN
    return x, y


def clip_scalar(val: int, v_min: int, v_max: int) -> int:
    """ convenience function to avoid using np.clip for scalar values """
    return v_min if val < v_min else v_max if val > v_max else val


class MultiLine(QGraphicsPathItem):
    def __init__(self, x: np.ndarray, y: np.ndarray, style: dict, group_number: int = 0) -> None:
        """x and y are 2D arrays of shape (N plots, N samples)"""
        self.x = x
        self.y = y
        self._x = x
        self._y = y
        connect = np.ones(x.shape, dtype=bool)
        connect[:, -1] = 0  # don't draw the segment between each trace
        self.path = arrayToQPath(x.flatten(), y.flatten(), connect.flatten(), finiteCheck=False)
        QGraphicsPathItem.__init__(self, self.path)
        self._originalPath = self.path  # will hold a PlotDataset for the original data
        self._modifiedPath = None  # will hold a PlotDataset for data after mapping transforms (e.g. log scale)
        self._datasetDisplay = None  # will hold a PlotDataset for data downsampled and limited for display
        self.opts = {
            'alphaHint': 1.0,
            'alphaMode': False,
            'downsample': 1,
            'autoDownsample': False,
            'downsampleMethod': 'peak',
            'skipFiniteCheck': True,
            'fftMode': False,
            'logMode': [False, False],
            'derivativeMode': False,
            'phasemapMode': False,
            'clipToView': False,
            'autoDownsampleFactor': 5.,  # draw ~5 samples per pixel,
            'stepMode': None
        }
        self.ints = ['plotData']
        if not isinstance(style, dict):
            self._style = {'color': QColor().fromRgb(random_rgb()),
                           'style': Qt.PenStyle.SolidLine,
                           'width': 1.0,
                           'fill': False,
                           'use_line_color': True,
                           'fill_color': QColor().fromRgb(random_rgb()),
                           'fill_opacity': 0.0}
        else:
            self._style = style.copy()
        color = self._style['color']
        color.setAlphaF(1.0)
        pen = mkPen(color=color, style=self._style['style'], width=self._style['width'])
        self.setPen(pen)

        self.group_number = group_number
        self._params = None

    def reset(self, path: QGraphicsPathItem) -> None:
        self.path = path
        self.setPath(path)
        color = self._style['color']
        color.setAlphaF(1.0)
        pen = mkPen(color=color, style=self._style['style'], width=self._style['width'])
        self.setPen(pen)

    def shape(self) -> QPainterPath:  # override because QGraphicsPathItem.shape is too expensive.
        return QGraphicsItem.shape(self)

    def boundingRect(self) -> QRectF:
        return self.path.boundingRect()

    def implements(self, interface: list[str | None] = None) -> bool:
        ints = self.ints
        if interface is None:
            return ints
        return interface in ints

    def name(self) -> str:
        """ Returns the name that represents this item in the legend. """
        return self.opts.get('name', None)

    def setAlpha(self, alpha: float, auto: bool) -> None:
        if self.opts['alphaHint'] == alpha and self.opts['alphaMode'] == auto:
            return
        self.opts['alphaHint'] = alpha
        self.opts['alphaMode'] = auto
        self.setOpacity(alpha)

    def get_style(self) -> dict:
        return self._style

    def get_group_number(self) -> int:
        return self.group_number

    def setDownsampling(self, ds: list[int | None] = None, auto: list[bool | None] = None,
                        method: list[str | None] = None) -> None:
        """
        Sets the downsampling mode of this item. Downsampling reduces the number
        of samples drawn to increase performance.

        ==============  =================================================================
        **Arguments:**
        ds              (int) Reduce visible plot samples by this factor. To disable,
                        set ds=1.
        auto            (bool) If True, automatically pick *ds* based on visible range
        mode            'subsample': Downsample by taking the first of N samples.
                        This method is fastest and least accurate.
                        'mean': Downsample by taking the mean of N samples.
                        'peak': Downsample by drawing a saw wave that follows the min
                        and max of the original data. This method produces the best
                        visual representation of the data but is slower.
        ==============  =================================================================
        """
        changed = False
        if ds is not None and self.opts['downsample'] != ds:
            changed = True
            self.opts['downsample'] = ds

        if auto is not None and self.opts['autoDownsample'] != auto:
            self.opts['autoDownsample'] = auto
            changed = True

        if method is not None and self.opts['downsampleMethod'] != method:
            changed = True
            self.opts['downsampleMethod'] = method

        if changed:
            self.updateItems(styleUpdate=False)

    def updateItems(self, styleUpdate: bool = True) -> None:
        # override styleUpdate request and always enforce update until we have a better solution for
        # - ScatterPlotItem losing per-point style information
        # - PlotDataItem performing multiple unnecessary setData calls on initialization

        curve_args = {}

        if styleUpdate:  # repeat style arguments only when changed
            for k, v in [
                ('pen', 'pen'),
                ('shadowPen', 'shadowPen'),
                ('fillLevel', 'fillLevel'),
                ('fillOutline', 'fillOutline'),
                ('fillBrush', 'brush'),
                ('antialias', 'antialias'),
                ('connect', 'connect'),
                ('stepMode', 'stepMode'),
                ('skipFiniteCheck', 'skipFiniteCheck')
            ]:
                if k in self.opts:
                    curve_args[v] = self.opts[k]

        p = self.getModifiedPath()
        self.reset(p)

    def getModifiedPath(self) -> QPainterPath:
        """
        Returns a :class:`PlotDataset <pyqtgraph.PlotDataset>` object that contains data suitable for display
        (after mapping and data reduction) as ``dataset.x`` and ``dataset.y``.
        Intended for internal use.
        """

        params = [self.opts['fftMode'],
                  self.opts['logMode'][0], self.opts['logMode'][1],
                  self.opts['derivativeMode'],
                  self.opts['phasemapMode'],
                  self.opts['downsample'], self.opts['autoDownsample'], self.opts['downsampleMethod'],
                  self.opts['autoDownsampleFactor'],
                  self.opts['clipToView']]
        if self._params and self._modifiedPath and params == self._params:
            return self._modifiedPath
        elif not any(params):
            return self._originalPath
        self._params = params
        x = self._x
        y = self._y
        # Apply data mapping functions if mapped dataset is not yet available:

        if self.opts['fftMode']:
            x, y = _fourierTransform(x, y)
            # Ignore the first bin for fft data if we have a logx scale
            if self.opts['logMode'][0]:
                x = x.T[1:].T
                y = y.T[1:].T

        if self.opts['derivativeMode']:  # plot dV/dt
            y = np.diff(y) / np.diff(x)
            x = x.T[:-1].T

        if self.opts['phasemapMode']:  # plot dV/dt vs V
            x = self._y.T[:-1].T
            y = np.diff(self._y) / np.diff(self._x)

        if True in self.opts['logMode']:
            x, y = apply_log_mapping(x, y, self.opts['logMode'])  # Apply log scaling for x and/or y-axis

        # apply processing that affects the on-screen display of data:
        ds = self.opts['downsample']
        if not isinstance(ds, int):
            ds = 1
        parent = self.parentItem()
        view = parent.getViewBox()
        if view is None:
            view_range = None
        else:
            view_range = view.viewRect()  # this is always up-to-date

        if self.opts['autoDownsample'] and view_range is not None and x.shape[0] > 1:
            # this option presumes that x-values have uniform spacing
            x_ = x[0]
            dx = float(x_[-1] - x_[0]) / (len(x_) - 1)
            if dx != 0.0:
                x0 = (view_range.left() - x_[0]) / dx
                x1 = (view_range.right() - x_[0]) / dx
                width = view.width()
                if width != 0.0:
                    ds = int(max(1, int((x1 - x0) / (width * self.opts['autoDownsampleFactor']))))
                #  downsampling is expensive; delay until after clipping.

        if self.opts['clipToView']:
            if view is None or view.autoRangeEnabled()[0]:
                pass  # no ViewBox to clip to, or view will autoscale to data range.
            else:
                # clip-to-view always presumes that x-values are in increasing order
                if view_range is not None and len(x[0]) > 1:
                    x_ = x[0]
                    # find first in-view value (left edge) and first out-of-view value (right edge)
                    # since we want the curve to go to the edge of the screen, we need to preserve
                    # one down-sampled point on the left and one of the right, so we extend the interval
                    x0 = np.searchsorted(x_, view_range.left()) - ds
                    x0 = clip_scalar(x0, 0, len(x_))  # workaround
                    # x0 = np.clip(x0, 0, len(x))

                    x1 = np.searchsorted(x_, view_range.right()) + ds
                    x1 = clip_scalar(x1, x0, len(x_))
                    # x1 = np.clip(x1, 0, len(x))
                    x = x.T[x0:x1].T
                    y = y.T[x0:x1].T

        if ds > 1:
            x, y = self.get_x_y_downsampled(x, y, ds)

        self.x = x
        self.y = y
        connect = np.ones(x.shape, dtype=bool)
        connect[:, -1] = 0  # don't draw the segment between each trace
        new_path = arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        self._modifiedPath = new_path
        return new_path

    def get_x_y_downsampled(self, x: np.ndarray, y: np.ndarray, ds: int) -> tuple[np.ndarray, np.ndarray]:
        if self.opts['downsampleMethod'] == 'subsample':
            x = x.T[::ds].T
            y = y.T[::ds].T
        elif self.opts['downsampleMethod'] == 'mean':
            n = x.shape[1] // ds
            m = x.shape[0]
            stx = ds // 2  # start of x-values; try to select a somewhat centered point
            x1 = x[0][stx:stx + n * ds:ds]
            x1 = np.tile(x1.T, m)
            x = x1.reshape(m, n)
            y_new = np.empty((m, n))
            for idx, i in enumerate(y):
                y_new[idx] = i[:n * ds].reshape(n, ds).mean(axis=1)
            y = y_new
        elif self.opts['downsampleMethod'] == 'peak':
            n = len(x[0]) // ds
            m = x.shape[0]
            x1 = np.empty((n, 2))
            stx = ds // 2  # start of x-values; try to select a somewhat centered point
            x1[:] = x[0][stx:stx + n * ds:ds, np.newaxis]
            x2 = x1.reshape(n * 2)
            x1 = np.tile(x2, m)
            x = x1.reshape(m, n * 2)
            y_new = np.empty((m, n * 2))
            for idx, i in enumerate(y):
                y1 = np.empty((n, 2))
                y2 = i[:n * ds].reshape((n, ds))
                y1[:, 0] = y2.max(axis=1)
                y1[:, 1] = y2.min(axis=1)
                y1 = y1.reshape(n * 2)
                y_new[idx] = y1
            y = y_new
        return x, y

    def setFftMode(self, state: bool) -> None:
        """
        ``state = True`` enables mapping the data by a fast Fourier transform.
        If the `x` values are not equidistant, the data set is resampled at
        equal intervals.
        """
        if self.opts['fftMode'] == state:
            return
        self.opts['fftMode'] = state
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def informViewBoundsChanged(self) -> None:
        """
        Inform this item's container ViewBox that the bounds of this item have changed.
        This is used by ViewBox to react if auto-range is enabled.
        """
        parent = self.parentItem()
        view = parent.getViewBox()
        if view is not None and hasattr(view, 'implements') and view.implements('ViewBox'):
            view.itemBoundsChanged(self)  # inform view, so it can update its range if it wants

    def setLogMode(self, xState: bool, yState: bool) -> None:
        """
        When log mode is enabled for the respective axis by setting ``xState`` or
        ``yState`` to `True`, a mapping according to ``mapped = np.log10( value )``
        is applied to the data. For negative or zero values, this results in a
        `NaN` value.
        """
        if self.opts['logMode'] == [xState, yState]:
            return
        self.opts['logMode'] = [xState, yState]
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setDerivativeMode(self, state: bool) -> None:
        """
        ``state = True`` enables derivative mode, where a mapping according to
        ``y_mapped = dy / dx`` is applied, with `dx` and `dy` representing the
        differences between adjacent `x` and `y` values.
        """
        if self.opts['derivativeMode'] == state:
            return
        self.opts['derivativeMode'] = state
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setPhasemapMode(self, state: bool) -> None:
        """
        ``state = True`` enables phase map mode, where a mapping
        according to ``x_mapped = y`` and ``y_mapped = dy / dx``
        is applied, plotting the numerical derivative of the data over the
        original `y` values.
        """
        if self.opts['phasemapMode'] == state:
            return
        self.opts['phasemapMode'] = state
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setClipToView(self, state: bool) -> None:
        """
        ``state=True`` enables clipping the displayed data set to the
        visible x-axis range.
        """
        if self.opts['clipToView'] == state:
            return
        self.opts['clipToView'] = state
        self.updateItems(styleUpdate=False)

    def getData(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the displayed data as the tuple (`xData`, `yData`) after mapping and data reduction.
        """
        y_t = self.y.T
        y_new = np.empty((1, self.y.shape[1])).T
        for i in range(self.y.shape[1]):
            y_new[i] = np.mean(y_t[i])
        x_r = self.x[0]
        y_r = y_new.T[0]
        return x_r, y_r


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
        # print(str(index.row()) + '----' + str(index.column()))
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

    def get_df_by_multiindex(self, mi: MultiIndex) -> DataFrame:
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

    def set_dataframe(self, df: pd.DataFrame) -> None:
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


class PandasModelInputTable(PandasModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: DataFrame):
        super().__init__(dataframe)
        self._dataframe = dataframe

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        # print(str(index.row()) + '----' + str(index.column()))
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
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def sort_values(self, current_name: str, ascending: bool) -> None:
        self._dataframe = self._dataframe.sort_values(by=current_name, ascending=ascending)
        self.modelReset.emit()

    def names_of_group(self, group_number: int) -> list[str]:
        return self._dataframe.loc[self._dataframe['Group'] == group_number].index

    def row_data_by_index(self, idx: str):
        return self._dataframe.loc[idx]


class PandasModelGroupsTable(PandasModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, parent: None, dataframe: DataFrame):
        super().__init__(dataframe)
        self._dataframe = dataframe
        self.parent = parent

    def get_group_name_by_int(self, group_number: int) -> str:
        return self._dataframe.loc[self._dataframe.index == group_number].iloc[0]['Group name']

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
        # print(str(index.row()) + '----' + str(index.column()))
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
        # print(str(index.row()) + '----' + str(index.column()))
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
        print('sort_by_border')
        self._dataframe = self._dataframe.sort_values(by=['Border'])
        self._dataframe.index = np.arange(1, len(self._dataframe) + 1)
        self.modelReset.emit()


class PandasModelDeconvLinesTable(PandasModel):
    """A model to interface a Qt view with pandas dataframe """

    sigCheckedChanged = Signal(int, bool)

    def __init__(self, parent: None, dataframe: DataFrame, checked: list[bool]):
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

    def __init__(self, parent: None, dataframe: DataFrame):
        super().__init__(dataframe)
        self.parent = parent
        self._dataframe = dataframe

    def append_row(self, line_index: int, param_name: str, param_value: float, min_v: float = None,
                   max_v: float = None, filename: str = '') -> None:
        # limits in self.parent.deconv_line_params_limits
        min_value = param_value
        max_value = param_value
        if param_name == 'a':
            min_value = 0
            max_value = param_value * 2
        elif param_name == 'x0':
            min_value = param_value - 1
            max_value = param_value + 1
        elif param_name == 'dx':
            min_value = np.round(param_value - param_value / np.pi, 5)
            max_value = np.round(param_value + param_value / np.pi, 5)
        elif param_name in self.parent.peak_shape_params_limits:
            min_value = self.parent.peak_shape_params_limits[param_name][0]
            max_value = self.parent.peak_shape_params_limits[param_name][1]
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

    def __init__(self, parent: None, dataframe: DataFrame):
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


class PandasModelBaselinedDataset(PandasModel):

    def __init__(self, parent: None, dataframe: DataFrame):
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


class PandasModelDeconvolutedDataset(PandasModel):

    def __init__(self, parent: None, dataframe: DataFrame):
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


class PandasModelPredictTable(PandasModel):

    def __init__(self, parent: None, dataframe: DataFrame):
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
        elif role == Qt.ItemDataRole.ForegroundRole and index.column() != 0:
            class_str = value.split(' ')[0]
            return self.parent.get_color_by_group_number(class_str)
        return None

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self._dataframe.iloc[index.row(), index.column()] = float(value)
            self.dataChanged.emit(index, index)
            return True

        else:
            return False


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
        self.RS.CommandDeconvLineDraggedAllowed = False
        command = CommandDeconvLineParameterChanged(self, self.RS, index, new_float, current_float, model,
                                                                 line_index, param_name, "Edit line %s" % index)
        self.RS.undoStack.push(command)

    @pyqtSlot()
    def editing_finished(self):
        self.commitData.emit(self.sender())


class IntervalsTableDelegate(QStyledItemDelegate):

    def __init__(self, rs):
        super().__init__()
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

    def __init__(self, line_types: list[str]):
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


class CurvePropertiesWindow(QWidget):
    sigStyleChanged = Signal(dict, dict, int)

    def __init__(self, parent: QMainWindow, style=None, idx: int = 0, fill_enabled: bool = True):
        super().__init__(parent, Qt.WindowType.Dialog)
        if style is None:
            style = {}
        self._style = style
        self.parent = parent
        self._idx = idx
        self.setFixedSize(240, 300)
        self._ui_form = curve_properties_ui.Ui_Dialog()
        self._ui_form.setupUi(self)
        self._init_style_cb()
        self._set_initial_values()
        self._ui_form.line_color_button.clicked.connect(self._set_new_line_color)
        self._ui_form.style_comboBox.currentTextChanged.connect(self._set_new_line_type)
        self._ui_form.width_doubleSpinBox.valueChanged.connect(self._set_new_width)
        self._ui_form.fill_group_box.toggled.connect(self._set_new_fill)
        self._ui_form.use_line_color_checkBox.toggled.connect(self._use_line_color_cb_toggled)
        self._ui_form.fill_color_button.clicked.connect(self._set_new_fill_color)
        self._ui_form.opacity_spinBox.valueChanged.connect(self._set_new_fill_opacity)
        self._ui_form.fill_group_box.setVisible(fill_enabled)

    def _init_style_cb(self) -> None:
        self._ui_form.style_comboBox.addItem('SolidLine')
        self._ui_form.style_comboBox.addItem('DotLine')
        self._ui_form.style_comboBox.addItem('DashLine')
        self._ui_form.style_comboBox.addItem('DashDotLine')
        self._ui_form.style_comboBox.addItem('DashDotDotLine')

    def _set_initial_values(self) -> None:
        if isinstance(self._style, dict):
            self._line_color_button_new_style_sheet(self._style['color'].name())
            self._select_style_cb_item(self._style['style'])
            self._ui_form.width_doubleSpinBox.setValue(self._style['width'])
            self._ui_form.fill_group_box.setChecked(self._style['fill'])
            self._ui_form.use_line_color_checkBox.setChecked(self._style['use_line_color'])
            self._fill_color_button_new_style_sheet(self._style['fill_color'].name())
            self._ui_form.opacity_spinBox.setValue(int(self._style['fill_opacity'] * 100))

    def _select_style_cb_item(self, pen_style: Qt.PenStyle) -> None:
        s = str(pen_style).split('.')[-1]
        self._ui_form.style_comboBox.setCurrentText(s)

    def _set_new_line_color(self):
        init_color = self._style['color']
        color_dialog = self.parent.color_dialog(init_color)
        color = color_dialog.getColor(init_color)
        if color.isValid():
            self._line_color_button_new_style_sheet(color.name())
            old_style = self._style.copy()
            self._style['color'] = color
            self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _line_color_button_new_style_sheet(self, hex_color: str) -> None:
        self._ui_form.line_color_button.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def _fill_color_button_new_style_sheet(self, hex_color: str) -> None:
        self._ui_form.fill_color_button.setStyleSheet(f"""*{{background-color: {hex_color};}}""")

    def _set_new_line_type(self, current_text: str) -> None:
        line_type = Qt.PenStyle.SolidLine
        match current_text:
            case 'DotLine':
                line_type = Qt.PenStyle.DotLine
            case 'DashLine':
                line_type = Qt.PenStyle.DashLine
            case 'DashDotLine':
                line_type = Qt.PenStyle.DashDotLine
            case 'DashDotDotLine':
                line_type = Qt.PenStyle.DashDotDotLine
        old_style = self._style.copy()
        self._style['style'] = line_type
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_width(self, f: float) -> None:
        old_style = self._style.copy()
        self._style['width'] = f
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_fill(self, b: bool) -> None:
        old_style = self._style.copy()
        self._style['fill'] = b
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _use_line_color_cb_toggled(self, b: bool) -> None:
        old_style = self._style.copy()
        self._style['use_line_color'] = b
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_fill_color(self):
        init_color = self._style['fill_color']
        color_dialog = self.parent.color_dialog(init_color)
        color = color_dialog.getColor(init_color)
        if color.isValid():
            self._fill_color_button_new_style_sheet(color.name())
            old_style = self._style.copy()
            self._style['fill_color'] = color
            self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def _set_new_fill_opacity(self, i: int) -> None:
        old_style = self._style.copy()
        self._style['fill_opacity'] = float(i / 100)
        self.sigStyleChanged.emit(self._style.copy(), old_style, self._idx)

    def accept(self) -> None:
        pass

    def reject(self) -> None:
        self.close()

    def idx(self) -> int:
        return self._idx


class DialogListBox(QDialog):
    def __init__(self, title: str, checked_ranges: list[int, int]) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(170, 100)
        self.ranges = QComboBox()
        for i in checked_ranges:
            self.ranges.addItem(str(i), i)
        button = QPushButton("OK", self)
        button.clicked.connect(self.ok_button_clicked)
        d_frame = QFrame(self)
        form_layout = QFormLayout(d_frame)
        form_layout.addRow(self.ranges)
        form_layout.addRow(button)

    def get_result(self) -> tuple[int, int]:
        return self.ranges.currentData()

    def ok_button_clicked(self) -> None:
        self.accept()


class SettingsDialog(QWidget):
    """form with program global settings. Open with settingsBtn"""

    def __init__(self, parent: QMainWindow):
        super().__init__(parent, Qt.WindowType.Dialog)
        self.parent = parent
        self.ui_form = Ui_Form()
        self.ui_form.setupUi(self)
        self.ui_form.tabWidget.setTabEnabled(1, False)
        self.parent.add_menu_combobox(self.ui_form.ThemeComboBox_Bckgrnd)
        self.parent.add_menu_combobox(self.ui_form.ThemeComboBox_Color, False)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowTitle('Preferences')
        self.ui_form.ThemeComboBox_Bckgrnd.currentTextChanged.connect(parent.theme_bckgrnd_text_changed)
        self.ui_form.ThemeComboBox_Color.currentTextChanged.connect(parent.theme_color_setting_text_changed)
        self.ui_form.ThemeComboBox_Bckgrnd.setCurrentText(parent.theme_bckgrnd)
        self.ui_form.ThemeComboBox_Color.setCurrentText(parent.theme_color)
        self.ui_form.recent_limit_spinBox.setValue(int(parent.recent_limit))
        self.ui_form.recent_limit_spinBox.valueChanged.connect(parent.recent_limit_spin_box_changed)
        self.ui_form.undo_limit_spinBox.setValue(int(parent.undoStack.undoLimit()))
        self.ui_form.undo_limit_spinBox.valueChanged.connect(parent.undo_limit_spin_box_changed)
        self.ui_form.auto_save_spinBox.setValue(parent.auto_save_minutes)
        self.ui_form.auto_save_spinBox.valueChanged.connect(parent.auto_save_spin_box_changed)
        self.ui_form.axis_font_size_spinBox.setValue(parent.plot_font_size)
        self.ui_form.axis_font_size_spinBox.valueChanged.connect(parent.axis_font_size_spin_box_changed)
        self.ui_form.axis_label_font_size_spinBox.setValue(int(parent.axis_label_font_size))
        self.ui_form.axis_label_font_size_spinBox.valueChanged.connect(parent.axis_label_font_size_changed)
        self.ui_form.auto_save_spinBox.setVisible(False)
        self.ui_form.auto_save_label.setVisible(False)
        self.closeEvent = self.settings_form_close_event

    def settings_form_close_event(self, _: QCloseEvent) -> None:
        theme_bckgrnd_ch = str(self.ui_form.ThemeComboBox_Bckgrnd.currentText())
        theme_color_ch = str(self.ui_form.ThemeComboBox_Color.currentText())
        recent_limit_ch = str(self.ui_form.recent_limit_spinBox.value())
        undo_stack_limit_ch = str(self.parent.undoStack.undoLimit())
        auto_save_minutes_ch = str(self.parent.auto_save_minutes)
        plot_font_size_ch = str(self.ui_form.axis_font_size_spinBox.value())
        axis_label_font_size_ch = str(self.ui_form.axis_label_font_size_spinBox.value())
        f = Path('preferences.txt').open('w+')
        f.write(
            theme_bckgrnd_ch + '\n' + theme_color_ch + '\n' + recent_limit_ch + '\n'
            + undo_stack_limit_ch + '\n' + auto_save_minutes_ch + '\n' + plot_font_size_ch + '\n'
            + axis_label_font_size_ch + '\n')
        f.close()
        self.close()



