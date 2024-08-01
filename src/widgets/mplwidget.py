# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Module for integrating Matplotlib with Qt via custom QWidget and Canvas classes.

This module includes:
- `MplCanvas`: A Matplotlib canvas class for creating and managing a figure.
- `MplWidget`: A QWidget class that integrates `MplCanvas` with a navigation toolbar.

Imports
-------
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT
from matplotlib.figure import Figure
"""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT
from matplotlib.figure import Figure


# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    """
    Matplotlib canvas class to create and manage a figure.

    Inherits from `matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg` and integrates a Matplotlib
    `Figure`
    into a Qt widget. Provides methods to interact with the figure and its axes.

    Attributes
    ----------
    figure : matplotlib.figure.Figure
        The Matplotlib figure instance managed by this canvas.

    Methods
    -------
    gca()
        Returns the current Axes instance on the figure.
    sca(ax)
        Sets the current Axes instance on the figure.
    """
    def __init__(self):
        """
        Initializes an MplCanvas instance by creating a new Matplotlib figure
        and setting up the canvas with the figure.
        """
        super().__init__()
        self.figure = Figure()
        self.figure.add_subplot(111)
        Canvas.__init__(self, self.figure)
        Canvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

    def gca(self):
        """
        Get the current Axes instance on the figure.

        Returns
        -------
        matplotlib.axes.Axes
            The current Axes instance.
        """
        return self.figure.gca()

    def sca(self, ax):
        """
        Set the current Axes instance on the figure.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes instance to set as the current Axes.
        """
        self.figure = ax.get_figure()


class MplWidget(QWidget):
    """
    A QWidget class that integrates a Matplotlib canvas with a navigation toolbar.

    Provides a user interface widget that includes a `MplCanvas` and a `NavigationToolbar2QT`
    for interacting with the canvas.

    Attributes
    ----------
    canvas : MplCanvas
        The Matplotlib canvas instance embedded in the widget.
    vbl : QVBoxLayout
        The layout manager for arranging widgets in the MplWidget.
    nav_bar : NavigationToolbar2QT
        The navigation toolbar instance for interacting with the Matplotlib canvas.

    Methods
    -------
    set_vbl(canvas)
        Replaces the current canvas with a new one and updates the layout.
    reset_canvas()
        Resets the canvas and toolbar to their initial states.
    """
    def __init__(self, parent=None):
        """
        Initializes an MplWidget instance by creating a new `MplCanvas`,
        adding it to a `QVBoxLayout`, and setting up a `NavigationToolbar2QT`.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget of this MplWidget (default is None).
        """
        QWidget.__init__(self, parent)  # Inherit from QWidget
        self.canvas = MplCanvas()  # Create canvas object
        self.vbl = QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.nav_bar = NavigationToolbar2QT(self.canvas, self)
        self.vbl.addWidget(self.nav_bar)
        self.setLayout(self.vbl)

    def set_vbl(self, canvas):
        """
        Replaces the current canvas with a new one and updates the layout.

        Parameters
        ----------
        canvas : MplCanvas
            The new canvas to be set in the widget.
        """
        self.canvas.setVisible(False)
        self.nav_bar.setVisible(False)
        self.vbl.removeWidget(self.canvas)
        self.vbl.removeWidget(self.nav_bar)

        self.canvas = canvas
        self.nav_bar = NavigationToolbar2QT(self.canvas, self)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.nav_bar)

    def reset_canvas(self):
        """
        Resets the canvas and toolbar to their initial states by creating
        a new `MplCanvas` and resetting the layout.

        This method removes the current canvas and toolbar from the layout,
        creates new instances, and adds them back to the layout.
        """
        self.canvas.setVisible(False)
        self.nav_bar.setVisible(False)
        self.vbl.removeWidget(self.canvas)
        self.vbl.removeWidget(self.nav_bar)
        self.canvas = MplCanvas()  # Create canvas object
        self.vbl = QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.nav_bar = NavigationToolbar2QT(self.canvas, self)
        self.vbl.addWidget(self.nav_bar)
