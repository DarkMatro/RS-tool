from asyncio import gather
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os import environ

import numpy as np
import winsound
from asyncqtpy import asyncSlot
from pandas import DataFrame, concat

from modules.functions_classificators import clf_predict
from modules.undo_redo import CommandUpdateInterpolated


class PredictLogic:

    def __init__(self, parent):
        self.parent = parent
        self.is_production_project = False
        self.interp_ref_array = None
        self.stat_models = {}
        self.y_axis_ref_EMSC = None

    @asyncSlot()
    async def do_predict_production(self) -> None:
        main_window = self.parent
        main_window.time_start = datetime.now()
        filenames = list(main_window.ImportedArray.keys())
        interpolated = await main_window.get_interpolated(filenames, self.interp_ref_array)
        if interpolated:
            command = CommandUpdateInterpolated(main_window, interpolated, "Interpolate files")
            main_window.undoStack.push(command)
        else:
            return
        await main_window.preprocessing.despike()
        await main_window.preprocessing.convert()
        await main_window.preprocessing.cut_first()
        await main_window.preprocessing.normalize()
        await main_window.preprocessing.smooth()
        if main_window.ui.dataset_type_cb.currentText() == 'Smoothed':
            await main_window.do_predict()
            main_window.set_modified(True)
            main_window.close_progress_bar()
            main_window.ui.statusBar.showMessage('Predicting completed (s)', 100000)
            winsound.MessageBeep()
            return
        await main_window.preprocessing.baseline_correction()
        if main_window.ui.dataset_type_cb.currentText() == 'Baseline corrected':
            await main_window.do_predict()
            main_window.set_modified(True)
            main_window.close_progress_bar()
            main_window.ui.statusBar.showMessage('Predicting completed (b.c.)', 100000)
            winsound.MessageBeep()
            return
        await main_window.preprocessing.trim()
        await main_window.batch_fit()
        df = main_window.ui.deconvoluted_dataset_table_view.model().dataframe().reset_index(drop=True)
        main_window.ui.deconvoluted_dataset_table_view.model().set_dataframe(df)
        await main_window.do_predict()
        main_window.set_modified(True)
        main_window.close_progress_bar()
        main_window.ui.statusBar.showMessage('Predicting completed (d)', 100000)
        winsound.MessageBeep()

    @asyncSlot()
    async def do_predict(self) -> None:
        main_window = self.parent
        main_window.time_start = datetime.now()
        main_window.ui.statusBar.showMessage('Predicting...')
        main_window.close_progress_bar()
        latest_stat_result = main_window.stat_analysis_logic.latest_stat_result
        if self.is_production_project:
            clfs = list(self.stat_models.keys())
        else:
            clfs = list(latest_stat_result.keys())
        if 'PCA' in clfs:
            clfs.remove('PCA')
        if 'PLS-DA' in clfs:
            clfs.remove('PLS-DA')
        if len(clfs) == 0:
            return
        main_window.open_progress_bar(max_value=len(clfs))
        main_window.open_progress_dialog("Predicting...", "Cancel", maximum=len(clfs))

        X, _, _, _, filenames = main_window.stat_analysis_logic.dataset_for_ml()
        executor = ThreadPoolExecutor()
        # Для БОльших датасетов возможно лучше будет ProcessPoolExecutor. Но таких пока нет
        main_window.current_executor = executor
        with executor:
            if self.is_production_project:
                main_window.current_futures = [main_window.loop.run_in_executor(executor, clf_predict, X,
                                                                                self.stat_models[i], i)
                                               for i in clfs]
            else:
                main_window.current_futures = [main_window.loop.run_in_executor(executor, clf_predict, X,
                                                                                latest_stat_result[i][
                                                                                    'model'], i)
                                               for i in clfs]
            for future in main_window.current_futures:
                future.add_done_callback(main_window.progress_indicator)
            result = await gather(*main_window.current_futures)
        if main_window.currentProgress.wasCanceled() or environ['CANCEL'] == '1':
            main_window.close_progress_bar()
            main_window.ui.statusBar.showMessage('Fitting cancelled.')
            return
        if result:
            self.update_predict_table(result, clfs, filenames)
        main_window.close_progress_bar()
        main_window.ui.statusBar.showMessage('Predicting completed', 10000)

    def update_predict_table(self, predict_data, classificators_names: list[str], filenames: list[str]) \
            -> None:
        """
        Update predict table at page 5
        @param predict_data: list[dict] - 'predicted', 'predicted_proba', 'clf_name'
        @param classificators_names: list[str]
        @param filenames: list[str]
        @return: None
        """
        df = DataFrame({'Filename': filenames})
        df2 = DataFrame(columns=classificators_names)
        for clf_result in predict_data:
            clf_name = clf_result['clf_name']
            predicted = clf_result['predicted']
            predicted_proba = np.round(clf_result['predicted_proba'] * 100., 0)
            str_list = []
            for i, pr in enumerate(predicted):
                if clf_name == 'XGBoost':
                    pr = self.parent.stat_analysis_logic.get_old_class_label(pr)
                class_proba = str(pr) + ' ' + str(predicted_proba[i])
                str_list.append(class_proba)
            df2[clf_name] = str_list
        df = df.reset_index(drop=True)
        df_final = concat([df, df2], axis=1)

        self.parent.ui.predict_table_view.model().set_dataframe(df_final)
