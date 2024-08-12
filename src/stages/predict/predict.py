# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Module for prediction and updating of prediction tables using Qt-based GUI.

This module provides the `Predict` class, which handles prediction functionalities
and updates prediction tables in a Qt-based GUI application. It includes methods for
reading, loading, resetting, and performing predictions, as well as updating the
prediction table in the user interface.
"""
import numpy as np
import pandas as pd
import winsound
from asyncqtpy import asyncSlot
from qtpy.QtCore import QObject

from src import get_parent, get_config
from src.pandas_tables import PandasModelPredictTable
from src.stages import clf_predict


class Predict(QObject):
    """
    Class to handle predictions and update prediction tables in a Qt-based GUI.

    This class manages the prediction workflow and updates the user interface with
    prediction results. It includes methods for resetting internal state, setting up
    the UI, reading and loading attributes, performing predictions, and updating
    prediction tables.

    Attributes
    ----------
    parent : QObject
        The parent object in the Qt-based application.
    is_production_project : bool
        Flag indicating if the project is in production mode.
    interp_ref_x_axis : Optional[any]
        Reference for the interpolation X-axis.
    y_axis_ref_emsc : Optional[any]
        Reference for the Y-axis EMSC.
    """
    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the Predict class.

        Parameters
        ----------
        parent : QObject
            The parent object in the Qt-based application.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.is_production_project = False
        self.interp_ref_x_axis = None
        self.y_axis_ref_emsc = None
        self.reset()
        self._set_ui()

    def reset(self):
        """
        Reset internal state and UI components related to predictions.

        This method clears the prediction dataset table and resets internal state attributes.
        """
        self._reset_predict_dataset_table()
        self.is_production_project = False
        self.interp_ref_x_axis = None
        self.y_axis_ref_emsc = None

    def _set_ui(self):
        """
        Set up UI components related to predictions.

        This method connects UI elements to their corresponding slots and configures
        the visibility of certain UI components.
        """
        mw = get_parent(self.parent, "MainWindow")
        mw.ui.page5_predict.clicked.connect(self.predict)
        mw.ui.predict_table_view.verticalScrollBar().valueChanged.connect(mw.move_side_scrollbar)
        mw.ui.predict_table_view.verticalHeader().setVisible(False)

    def read(self, production_export: bool=False) -> dict:
        """
        Read current attributes data.

        Parameters
        ----------
        production_export : bool, optional
            Flag indicating if the data is for a production export (default is False).

        Returns
        -------
        dict
            A dictionary containing current attributes data including 'predict_df',
            'y_axis_ref_emsc', 'interp_ref_x_axis', and 'is_production_project'.
        """
        mw = get_parent(self.parent, "MainWindow")
        dt = {"y_axis_ref_emsc": self.y_axis_ref_emsc,
              "interp_ref_x_axis": self.interp_ref_x_axis,
              'is_production_project': self.is_production_project
              }
        if not production_export:
            dt['predict_df'] = mw.ui.predict_table_view.model().dataframe
        else:
            dt['is_production_project'] = True
        return dt

    def load(self, db: dict):
        """
        Load attributes from a dictionary into the class.

        Parameters
        ----------
        db : dict
            A dictionary containing data to be loaded into the class attributes.
        """
        mw = get_parent(self.parent, "MainWindow")
        self.is_production_project = db['is_production_project']
        if 'predict_df' in db:
            mw.ui.predict_table_view.model().set_dataframe(db["predict_df"])
        self.y_axis_ref_emsc = db['y_axis_ref_emsc']
        self.interp_ref_x_axis = db['interp_ref_x_axis']

    def _reset_predict_dataset_table(self) -> None:
        """
        Reset the prediction dataset table in the UI.

        This method clears and resets the model of the prediction table view to an empty
        DataFrame with a 'Filename' column.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        df = pd.DataFrame(columns=["Filename"])
        model = PandasModelPredictTable(context, df)
        mw.ui.predict_table_view.setModel(model)

    @asyncSlot()
    async def predict(self):
        """
        Perform predictions based on the current project settings.

        This method invokes the appropriate prediction method based on the project type
        and performs necessary preprocessing steps.
        """
        if self.is_production_project:
            await self.do_predict_production()
        else:
            await self.do_predict()

    @asyncSlot()
    async def do_predict_production(self) -> None:
        """
        Perform production-specific prediction steps.

        This method handles preprocessing, conversion, and prediction steps specific to
        production projects. It also updates the prediction table and emits a beep sound
        upon completion.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        await context.preprocessing.stages.input_data.interpolate_clicked()
        if context.preprocessing.stages.input_data.ui.despike_gb.isChecked():
            await context.preprocessing.stages.input_data.despike_clicked()
        await context.preprocessing.stages.convert_data.convert_clicked()
        await context.preprocessing.stages.cut_data.cut_clicked()
        for w in mw.ui.drag_widget.get_widgets_order(False)[3:-2]:
            await w.process_clicked()
        await context.preprocessing.stages.trim_data.cut_clicked()
        await context.preprocessing.stages.av_data.average_clicked()

        if mw.ui.dataset_type_cb.currentText() == 'Smoothed':
            await self.do_predict()
            context.set_modified()
            winsound.MessageBeep()
            return
        if mw.ui.dataset_type_cb.currentText() == 'Baseline corrected':
            await self.do_predict()
            context.set_modified()
            winsound.MessageBeep()
            return
        await context.decomposition.b.batch_fit()
        await self.do_predict()
        context.set_modified()
        winsound.MessageBeep()

    @asyncSlot()
    async def do_predict(self) -> None:
        """
        Perform general prediction steps.

        This method handles the core prediction logic, including retrieving models,
        running predictions, and updating the prediction table. It shows progress
        information and displays status messages as needed.
        """
        mw = get_parent(self.parent, "MainWindow")
        cfg = get_config("texty")["predict"]
        ml = self.parent.ml
        latest_stat_result = ml.data
        clfs = list(latest_stat_result.keys())
        if 'PCA' in clfs:
            clfs.remove('PCA')
        if len(clfs) == 0:
            return
        mw.progress.open_progress(cfg, len(clfs))
        x, _, _, _, filenames = ml.dataset_for_ml()
        kwargs = {'x': x}
        models = [(v['model'], k) for k, v in ml.data.items() if k != 'PCA']
        result = await mw.progress.run_in_executor('predict', clf_predict, models, **kwargs)
        if mw.progress.close_progress(cfg):
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
            return
        self.update_predict_table(result, clfs, filenames)

    def update_predict_table(self, predict_data, clf_names: list[str],
                             filenames: list[str]) \
            -> None:
        """
        Update the prediction table in the UI.

        Parameters
        ----------
        predict_data : list[dict]
            A list of dictionaries containing prediction results. Each dictionary
            includes 'predicted', 'predicted_proba', and 'clf_name'.
        clf_names : list[str]
            A list of classifier names.
        filenames : list[str]
            A list of filenames corresponding to the predictions.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, "MainWindow")
        df = pd.DataFrame({'Filename': filenames})
        df2 = pd.DataFrame(columns=clf_names)
        for clf_result in predict_data:
            clf_name = clf_result['clf_name']
            predicted_proba = np.round(clf_result['y_score'] * 100.)
            str_list = []
            y_pred = clf_result['y_pred']
            if clf_name == 'XGBoost':
                y_pred = self.parent.ml.le.inverse_transform(clf_result['y_pred'])
            for i, pr in enumerate(y_pred):
                str_list.append(f"{pr} {predicted_proba[i]}")
            df2[clf_name] = str_list
        df = df.reset_index(drop=True)
        df_final = pd.concat([df, df2], axis=1)
        mw.ui.predict_table_view.model().set_dataframe(df_final)
