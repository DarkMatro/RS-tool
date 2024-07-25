from qtpy.QtWidgets import QHeaderView, QAbstractItemView
from pandas import DataFrame

from src import get_parent
from src.pandas_tables import PandasModelDeconvTable


class TableFilenames:
    def __init__(self, parent):
        self.parent = parent
        self.reset()
        self.set_ui()

    def reset(self):
        mw = get_parent(self.parent, "MainWindow")
        df = DataFrame(columns=["Filename"])
        model = PandasModelDeconvTable(df)
        mw.ui.dec_table.setModel(model)
        if mw.ui.input_table.model() is not None:
            df = mw.ui.input_table.model().dataframe()
            mw.ui.dec_table.model().concat_deconv_table(filename=df.index)

    def set_ui(self):
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.dec_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        mw.ui.dec_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        mw.ui.dec_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        mw.ui.dec_table.doubleClicked.connect(self.parent.dec_table_double_clicked)
