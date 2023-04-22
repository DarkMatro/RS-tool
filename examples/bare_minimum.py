import sys

from PySide6 import QtWidgets

from modules import apply_stylesheet

# from PySide2 import QtWidgets
# from PyQt5 import QtWidgets
# from PyQt6 import QtWidgets

# create the application and the main window
app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()

# setup stylesheet
apply_stylesheet(app, theme='dark_teal.xml')

# run
window.show()

if hasattr(app, 'exec'):
    app.exec()
else:
    app.exec_()

