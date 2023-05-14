from qtpy.QtWidgets import QGraphicsPathItem, QGraphicsItem
import numpy as np
from pyqtgraph import arrayToQPath, mkPen
from qtpy.QtGui import QColor, QPainterPath
from modules.static_functions import random_rgb
from qtpy.QtCore import Qt, QRectF
from warnings import catch_warnings, simplefilter


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
