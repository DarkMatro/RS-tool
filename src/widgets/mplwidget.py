from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas, NavigationToolbar2QT
from matplotlib.figure import Figure


# matplotlib.rcParams["toolbar"] = "toolmanager"
# plt.style.use(['dark_background'])
# Ensure using PyQt5 backend
# matplotlib.use('QT5Agg')


# Matplotlib canvas class to create figure
class MplCanvas(Canvas):
    def __init__(self):
        super().__init__()
        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        Canvas.__init__(self, self.figure)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

    def gca(self):
        return self.figure.gca()


# Matplotlib widget
class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)  # Inherit from QWidget
        self.canvas = MplCanvas()  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.nav_bar = NavigationToolbar2QT(self.canvas, self)
        self.vbl.addWidget(self.nav_bar)
        self.setLayout(self.vbl)

    def set_vbl(self, canvas):
        self.canvas.setVisible(False)
        self.nav_bar.setVisible(False)
        self.vbl.removeWidget(self.canvas)
        self.vbl.removeWidget(self.nav_bar)

        self.canvas = canvas
        self.nav_bar = NavigationToolbar2QT(self.canvas, self)
        self.vbl.addWidget(self.canvas)
        self.vbl.addWidget(self.nav_bar)

    def reset_canvas(self):
        self.canvas.setVisible(False)
        self.nav_bar.setVisible(False)
        self.vbl.removeWidget(self.canvas)
        self.vbl.removeWidget(self.nav_bar)
        self.canvas = MplCanvas()  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()  # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.nav_bar = NavigationToolbar2QT(self.canvas, self)
        self.vbl.addWidget(self.nav_bar)





