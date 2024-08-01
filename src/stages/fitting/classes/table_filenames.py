# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
This module provides functionality for managing the filenames table in a graphical user interface
for spectral analysis.

The main class, TableFilenames, is responsible for initializing, resetting, and setting up the user
interface for the filenames table. It integrates with the main window of the application, manages
the display of filenames, and handles user interactions with the table.
"""

from pandas import DataFrame
from qtpy.QtWidgets import QHeaderView, QAbstractItemView

from src import get_parent
from src.pandas_tables import PandasModelDeconvTable


class TableFilenames:
    """
    A class to manage the filenames table in a GUI for spectral analysis.

    This class is responsible for initializing, resetting, and setting up the user interface for
    the filenames table.
    It integrates with the main window of the application, manages the display of filenames, and
    handles user interactions with the table.

    Parameters
    ----------
    parent : object
        The parent object, typically the main window of the application.
    """
    def __init__(self, parent):
        """
        Initialize the TableFilenames instance.

        This method sets up the initial state of the filenames table and the user interface.

        Parameters
        ----------
        parent : object
            The parent object, typically the main window of the application.
        """
        self.parent = parent
        self.reset()
        self.set_ui()

    def reset(self):
        """
        Reset the filenames table.

        This method clears the filenames table and sets up a new empty model. It also populates the
        table with filenames from the input table if available.
        """
        mw = get_parent(self.parent, "MainWindow")
        df = DataFrame(columns=["Filename"])
        model = PandasModelDeconvTable(df)
        mw.ui.dec_table.setModel(model)
        if mw.ui.input_table.model() is not None:
            df = mw.ui.input_table.model().dataframe
            mw.ui.dec_table.model().concat_deconv_table(filename=df.index)

    def set_ui(self):
        """
        Set up the user interface for the filenames table.

        This method configures the table headers, selection behavior, and connects the double click
        event to the appropriate handler in the parent.
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.dec_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        mw.ui.dec_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        mw.ui.dec_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        mw.ui.dec_table.doubleClicked.connect(self.parent.dec_table_double_clicked)
