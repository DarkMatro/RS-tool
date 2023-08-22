import copy
from asyncio import create_task, gather, sleep
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from gc import collect
from logging import info, debug
import torch
from sklearn.model_selection import GridSearchCV
import lmfit.model
import numpy as np
import pandas as pd
import shap
from asyncqtpy import asyncSlot
from lmfit.model import ModelResult
from pyqtgraph import mkPen, ROI
from qtpy.QtCore import QModelIndex, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QUndoCommand, QStyledItemDelegate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from modules.functions_classificators import model_metrics
from modules.static_functions import random_rgb, random_line_style, curve_pen_brush_by_style, set_roi_size_pos, \
    calculate_vips


class CommandImportFiles(QUndoCommand):
    """
    change dict  self.ImportedArray
    add rows to input table
    update input plot
    input_result is: 0 - name, 1 - array, 2 - group number, 3 - min_nm, 4 - max_nm, 5 - fwhm
    """

    def __init__(self, rs, input_result: list[tuple[str, np.ndarray, str, str, str, float]],
                 description: str) -> None:
        super(CommandImportFiles, self).__init__(description)
        self.RS = rs
        self.input_result = input_result
        self.deconv_table = rs.ui.dec_table
        self.setText(description)
        self.input_table = rs.ui.input_table
        self.before = []
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop
        for i in rs.ImportedArray.items():
            self.before.append((i[0], i[1]))

    @asyncSlot()
    async def redo(self) -> None:
        self.RS.time_start = datetime.now() - (datetime.now() - self.RS.time_start)
        self.RS.ui.statusBar.showMessage('Redo...' + self.text())
        self.RS.disable_buttons(True)
        columns = list(zip(*self.input_result))
        self.input_table.model().append_row_input_table(name=columns[0], min_nm=columns[3], max_nm=columns[4],
                                                        group=columns[2], despiked_nm=[''], rayleigh_line=[''],
                                                        fwhm=columns[5])
        self.deconv_table.model().append_row_deconv_table(filename=columns[0])
        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self.import_redo, i)
                       for i in self.input_result]
            await gather(*futures)
        await self.RS.preprocessing.update_plot_item(self.RS.ImportedArray.items())
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        self.RS.time_start = datetime.now() - (datetime.now() - self.RS.time_start)
        self.RS.ui.statusBar.showMessage('Undo...' + self.text())
        self.RS.disable_buttons(True)
        columns = list(zip(*self.input_result))
        self.input_table.model().delete_rows_input_table(names=columns[0])
        self.deconv_table.model().delete_rows_deconv_table(names=columns[0])
        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self.import_undo, i)
                       for i in self.input_result]
            await gather(*futures)
        await self.RS.preprocessing.update_plot_item(self.RS.ImportedArray.items())
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    def import_redo(self, i: tuple[str, np.ndarray, int, float, float, float]) -> None:
        basename = i[0]
        n_array = i[1]
        self.RS.ImportedArray[basename] = n_array

    def import_undo(self, i: tuple[str, np.ndarray, str, float, float, float]) -> None:
        basename = i[0]
        del self.RS.ImportedArray[basename]

    async def _stop(self) -> None:
        self.input_table.move(0, 1)
        self.RS.disable_buttons(False)
        self.update_undo_redo_tooltips()
        self.RS.set_buttons_ability()
        time_end = self.RS.time_start
        if not self.RS.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.RS.time_start = None
        self.RS.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.RS.set_modified()
        self.input_table.move(0, -1)
        self.RS.decide_vertical_scroll_bar_visible()
        await sleep(0)


class CommandAddGroup(QUndoCommand):
    """добавляет строку в таблицу групп, обновляются цвета графиков, перестраиваются все графики"""

    def __init__(self, rs, row: int, color: QColor, description: str) -> None:
        super(CommandAddGroup, self).__init__(description)
        self.setText(description)
        self.GroupsTable = rs.ui.GroupsTable
        self.row = row
        self.color = color
        self._style = {'color': color,
                       'style': Qt.PenStyle.SolidLine,
                       'width': 1.0,
                       'fill': False,
                       'use_line_color': True,
                       'fill_color': QColor().fromRgb(random_rgb()),
                       'fill_opacity': 0.0}
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.RS = rs

    def redo(self) -> None:
        self.GroupsTable.model().append_group(group='Group ' + str(self.row + 1),
                                              style=self._style, index=self.row + 1)
        #  Перестраиваем все графики, потому что потому
        self.RS.preprocessing.update_averaged()
        self.RS.update_all_plots()
        self.update_undo_redo_tooltips()

    def undo(self) -> None:
        self.GroupsTable.model().remove_group(self.row + 1)
        #  Перестраиваем все графики, потому что потому
        self.RS.preprocessing.update_averaged()
        self.RS.update_all_plots()
        self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')

        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.RS.set_modified()


class CommandDeleteGroup(QUndoCommand):

    def __init__(self, rs, row: int, name: str, style: dict, description: str) -> None:
        super(CommandDeleteGroup, self).__init__(description)
        self.setText(description)
        self.GroupsTable = rs.ui.GroupsTable
        self.row = row
        self.style = style
        self.name = name
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.RS = rs

    def redo(self) -> None:
        self.GroupsTable.model().remove_group(self.row + 1)
        self.RS.preprocessing.update_averaged()
        self.RS.update_all_plots()
        self.update_undo_redo_tooltips()

    def undo(self) -> None:
        self.GroupsTable.model().append_group(group='Group ' + str(self.row + 1),
                                              style=self.style, index=self.row + 1)
        self.RS.preprocessing.update_averaged()
        self.RS.update_all_plots()
        self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')

        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.RS.set_modified()


class CommandChangeGroupCell(QUndoCommand):
    """ for changing group number of 1 row in input_table
    Parameters
    ______________________________
    item: input_table.Item()
    """

    def __init__(self, rs, index: QModelIndex, new_value: str, description: str) -> None:
        super(CommandChangeGroupCell, self).__init__(description)
        self.setText(description)
        self.rs = rs
        self.index = index
        self.previous_value = int(rs.previous_group_of_item)
        self.new_value = int(new_value)
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack

    def redo(self) -> None:
        self.rs.ui.input_table.model().change_cell_data(self.index.row(), self.index.column(), self.new_value)
        filename = self.rs.ui.input_table.model().index_by_row(self.index.row())
        if self.rs.ui.smoothed_dataset_table_view.model().rowCount() > 0:
            idx_sm = self.rs.ui.smoothed_dataset_table_view.model().idx_by_column_value('Filename', filename)
            self.rs.ui.smoothed_dataset_table_view.model().set_cell_data_by_idx_col_name(idx_sm, 'Class',
                                                                                         self.new_value)
        if self.rs.ui.baselined_dataset_table_view.model().rowCount() > 0:
            idx_bl = self.rs.ui.baselined_dataset_table_view.model().idx_by_column_value('Filename', filename)
            self.rs.ui.baselined_dataset_table_view.model().set_cell_data_by_idx_col_name(idx_bl, 'Class',
                                                                                          self.new_value)
        if self.rs.ui.deconvoluted_dataset_table_view.model().rowCount() > 0:
            idx_dc = self.rs.ui.deconvoluted_dataset_table_view.model().idx_by_column_value('Filename', filename)
            self.rs.ui.deconvoluted_dataset_table_view.model().set_cell_data_by_idx_col_name(idx_dc, 'Class',
                                                                                             self.new_value)
        self.rs.preprocessing.update_averaged()
        self.rs.update_all_plots()
        self.update_undo_redo_tooltips()

    def undo(self) -> None:
        self.rs.ui.input_table.model().setData(self.index.row(), self.index.column(), self.previous_value)
        filename = self.rs.ui.input_table.model().index_by_row(self.index.row())
        idx_sm = self.rs.ui.smoothed_dataset_table_view.model().idx_by_column_value('Filename', filename)
        self.rs.ui.smoothed_dataset_table_view.model().set_cell_data_by_idx_col_name(idx_sm, 'Class',
                                                                                     self.previous_value)
        idx_bl = self.rs.ui.baselined_dataset_table_view.model().idx_by_column_value('Filename', filename)
        self.rs.ui.baselined_dataset_table_view.model().set_cell_data_by_idx_col_name(idx_bl, 'Class',
                                                                                      self.previous_value)
        idx_dc = self.rs.ui.deconvoluted_dataset_table_view.model().idx_by_column_value('Filename', filename)
        self.rs.ui.deconvoluted_dataset_table_view.model().set_cell_data_by_idx_col_name(idx_dc, 'Class',
                                                                                         self.previous_value)
        self.rs.preprocessing.update_averaged()
        self.rs.update_all_plots()
        self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.rs.set_modified()


class CommandChangeGroupCellsBatch(QUndoCommand):
    """ меняет группу для выделенных строк и цвет на графике
    Parameters:
    ___________________________
    undo_dict: undo_dict[filename] = (new_value, old_value)
     """

    def __init__(self, rs, undo_dict: dict[int, tuple[int, int]], description: str) -> None:
        super(CommandChangeGroupCellsBatch, self).__init__(description)
        self.setText(description)
        self.input_table = rs.ui.input_table
        self.undo_dict = undo_dict
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.rs = rs
        self.loop = rs.loop

    @asyncSlot()
    async def redo(self) -> None:
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)
        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self._do, i, False)
                       for i in self.undo_dict.items()]
            await gather(*futures)
        await self.rs.preprocessing.update_averaged()
        await self.rs.update_all_plots()
        self.rs.disable_buttons(False)
        self.update_undo_redo_tooltips()

    @asyncSlot()
    async def undo(self) -> None:
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)
        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self._do, i, True)
                       for i in self.undo_dict.items()]
            await gather(*futures)
        await self.rs.preprocessing.update_averaged()
        await self.rs.update_all_plots()
        self.rs.disable_buttons(False)
        self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.rs.set_modified()

    def _do(self, item: tuple[int, tuple[int, int]], undo: bool = False) -> None:
        row = item[0]
        new_value = item[1][0]
        old_value = item[1][1]
        if undo:
            group_number = int(old_value)
        else:
            group_number = int(new_value)
        self.input_table.model().change_cell_data(row, 2, group_number)
        filename = self.input_table.model().index_by_row(row)
        if self.rs.ui.smoothed_dataset_table_view.model().rowCount() > 0 \
                and filename in self.rs.ui.smoothed_dataset_table_view.model().filenames:
            idx_sm = self.rs.ui.smoothed_dataset_table_view.model().idx_by_column_value('Filename', filename)
            self.rs.ui.smoothed_dataset_table_view.model().set_cell_data_by_idx_col_name(idx_sm, 'Class', group_number)
        if self.rs.ui.baselined_dataset_table_view.model().rowCount() > 0 \
                and filename in self.rs.ui.baselined_dataset_table_view.model().filenames:
            idx_bl = self.rs.ui.baselined_dataset_table_view.model().idx_by_column_value('Filename', filename)
            self.rs.ui.baselined_dataset_table_view.model().set_cell_data_by_idx_col_name(idx_bl, 'Class', group_number)
        if self.rs.ui.deconvoluted_dataset_table_view.model().rowCount() > 0 \
                and filename in self.rs.ui.deconvoluted_dataset_table_view.model().filenames:
            idx_dc = self.rs.ui.deconvoluted_dataset_table_view.model().idx_by_column_value('Filename', filename)
            self.rs.ui.deconvoluted_dataset_table_view.model().set_cell_data_by_idx_col_name(idx_dc, 'Class',
                                                                                             group_number)


class CommandDeleteInputSpectrum(QUndoCommand):
    """ for delete/undo_add rows in input_table, all dicts and input plot, rebuild plots"""

    def __init__(self, rs, description: str) -> None:
        super(CommandDeleteInputSpectrum, self).__init__(description)
        self.setText(description)
        self.input_table = rs.ui.input_table
        self.selected_indexes = self.input_table.selectionModel().selectedIndexes()
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.dict = dict()
        self.rs = rs
        self.dictArray = dict()
        self.BeforeDespikeArray = dict()
        self.LaserPeakFWHM_old = dict()
        self.ConvertedDict_old = dict()
        self.CuttedFirstDict_old = dict()
        self.NormalizedDict_old = dict()
        self.SmoothedDict_old = dict()
        self.BaselineDict_old = dict()
        self.BaselineCorrectedDict_old = dict()
        self.report_result_old = dict()
        self.sigma3_old = dict()
        self.smoothed_df_old = rs.ui.smoothed_dataset_table_view.model().dataframe()
        self.baselined_df_old = rs.ui.baselined_dataset_table_view.model().dataframe()
        self.deconvoluted_df_old = rs.ui.deconvoluted_dataset_table_view.model().dataframe()
        self.prepare()
        self.df = rs.ui.fit_params_table.model().query_result_with_list('filename == @input_list',
                                                                        list(self.dict.keys()))
        self.loop = rs.loop

    def prepare(self) -> None:
        for i in self.selected_indexes:
            row_data = self.input_table.model().row_data(i.row())
            filename = row_data.name
            self.dict[filename] = row_data
            self.dictArray[filename] = self.rs.ImportedArray[filename]
            if self.rs.preprocessing.BeforeDespike and filename in self.rs.preprocessing.BeforeDespike:
                self.BeforeDespikeArray[filename] = self.rs.preprocessing.BeforeDespike[filename]
            if self.rs.preprocessing.ConvertedDict and filename in self.rs.preprocessing.ConvertedDict:
                self.ConvertedDict_old[filename] = self.rs.preprocessing.ConvertedDict[filename]
            if self.rs.preprocessing.CuttedFirstDict and filename in self.rs.preprocessing.CuttedFirstDict:
                self.CuttedFirstDict_old[filename] = self.rs.preprocessing.CuttedFirstDict[filename]
            if self.rs.preprocessing.NormalizedDict and filename in self.rs.preprocessing.NormalizedDict:
                self.NormalizedDict_old[filename] = self.rs.preprocessing.NormalizedDict[filename]
            if self.rs.preprocessing.smoothed_spectra and filename in self.rs.preprocessing.smoothed_spectra:
                self.SmoothedDict_old[filename] = self.rs.preprocessing.smoothed_spectra[filename]
            if self.rs.preprocessing.baseline_dict and filename in self.rs.preprocessing.baseline_dict:
                self.BaselineDict_old[filename] = self.rs.preprocessing.baseline_dict[filename]
            if self.rs.preprocessing.baseline_corrected_dict and filename in self.rs.preprocessing.baseline_corrected_dict:
                self.BaselineCorrectedDict_old[filename] = self.rs.preprocessing.baseline_corrected_dict[filename]
            if self.rs.fitting.report_result and filename in self.rs.report_result:
                self.report_result_old[filename] = self.rs.fitting.report_result[filename]
            if self.rs.fitting.sigma3 and filename in self.rs.fitting.sigma3:
                self.sigma3_old[filename] = self.rs.fitting.sigma3[filename]

    @asyncSlot()
    async def redo(self) -> None:
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)
        self.input_table.model().delete_rows_input_table(self.dict.keys())
        self.rs.ui.dec_table.model().delete_rows_deconv_table(self.dict.keys())
        self.rs.ui.fit_params_table.model().delete_rows_by_filenames(list(self.dict.keys()))
        if self.dict:
            with ThreadPoolExecutor() as executor:
                futures = [self.loop.run_in_executor(executor, self._del_row, i)
                           for i in self.dict.keys()]
                await gather(*futures)
            await self.rs.preprocessing.update_averaged()
            await self.rs.update_all_plots()
        self.rs.ui.smoothed_dataset_table_view.model().delete_rows_by_filenames(list(self.dict.keys()))
        self.rs.ui.smoothed_dataset_table_view.model().sort_index()
        self.rs.ui.baselined_dataset_table_view.model().delete_rows_by_filenames(list(self.dict.keys()))
        self.rs.ui.baselined_dataset_table_view.model().sort_index()
        self.rs.ui.deconvoluted_dataset_table_view.model().delete_rows_by_filenames(list(self.dict.keys()))
        self.rs.ui.deconvoluted_dataset_table_view.model().sort_index()
        create_task(self._stop_del())

    @asyncSlot()
    async def undo(self) -> None:
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)
        columns = list(zip(*self.dict.values()))
        self.input_table.model().append_row_input_table(name=self.dict.keys(), min_nm=columns[0], max_nm=columns[1],
                                                        group=columns[2], despiked_nm=columns[3],
                                                        rayleigh_line=columns[4], fwhm=columns[5])
        self.rs.ui.dec_table.model().append_row_deconv_table(self.dict.keys())
        self.rs.ui.fit_params_table.model().concat_df(self.df)
        if self.dict:
            with ThreadPoolExecutor() as executor:
                futures = [self.loop.run_in_executor(executor, self._add_row, i)
                           for i in self.dict.keys()]
                await gather(*futures)

            self.rs.disable_buttons(False)
            await self.rs.preprocessing.update_averaged()
            await self.rs.update_all_plots()
        self.rs.ui.smoothed_dataset_table_view.model().set_dataframe(self.smoothed_df_old)
        self.rs.ui.smoothed_dataset_table_view.model().sort_index()
        self.rs.ui.baselined_dataset_table_view.model().set_dataframe(self.baselined_df_old)
        self.rs.ui.baselined_dataset_table_view.model().sort_index()
        self.rs.ui.deconvoluted_dataset_table_view.model().set_dataframe(self.deconvoluted_df_old)
        self.rs.ui.deconvoluted_dataset_table_view.model().sort_index()
        create_task(self._stop_del())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.rs.set_buttons_ability()

    def _del_row(self, key: str) -> None:
        if key in self.rs.ImportedArray:
            del self.rs.ImportedArray[key]
        if key in self.rs.preprocessing.BeforeDespike:
            del self.rs.preprocessing.BeforeDespike[key]
        if key in self.rs.preprocessing.ConvertedDict:
            del self.rs.preprocessing.ConvertedDict[key]
        if key in self.rs.preprocessing.NormalizedDict:
            del self.rs.preprocessing.NormalizedDict[key]
        if key in self.rs.preprocessing.smoothed_spectra:
            del self.rs.preprocessing.smoothed_spectra[key]
        if key in self.rs.preprocessing.CuttedFirstDict:
            del self.rs.preprocessing.CuttedFirstDict[key]
        if key in self.rs.preprocessing.baseline_dict:
            del self.rs.preprocessing.baseline_dict[key]
        if key in self.rs.preprocessing.baseline_corrected_dict:
            del self.rs.preprocessing.baseline_corrected_dict[key]
        if key in self.rs.fitting.report_result:
            del self.rs.fitting.report_result[key]
        if key in self.rs.fitting.sigma3:
            del self.rs.fitting.sigma3[key]

    def _add_row(self, key: str) -> None:
        n_array = self.dictArray[key]
        self.rs.ImportedArray[key] = n_array
        if key in self.BeforeDespikeArray:
            self.rs.preprocessing.BeforeDespike[key] = self.BeforeDespikeArray[key]
        if key in self.ConvertedDict_old:
            self.rs.preprocessing.ConvertedDict[key] = self.ConvertedDict_old[key]
        if key in self.CuttedFirstDict_old:
            self.rs.preprocessing.CuttedFirstDict[key] = self.CuttedFirstDict_old[key]
        if key in self.NormalizedDict_old:
            self.rs.preprocessing.NormalizedDict[key] = self.NormalizedDict_old[key]
        if key in self.SmoothedDict_old:
            self.rs.preprocessing.smoothed_spectra[key] = self.SmoothedDict_old[key]
        if key in self.BaselineDict_old:
            self.rs.preprocessing.baseline_dict[key] = self.BaselineDict_old[key]
        if key in self.BaselineCorrectedDict_old:
            self.rs.preprocessing.baseline_corrected_dict[key] = self.BaselineCorrectedDict_old[key]
        if key in self.report_result_old:
            self.rs.fitting.report_result[key] = self.report_result_old[key]
        if key in self.sigma3_old:
            self.rs.fitting.sigma3[key] = self.sigma3_old[key]

    async def _stop_del(self) -> None:
        self.rs.disable_buttons(False)
        self.rs.decide_vertical_scroll_bar_visible()
        self.update_undo_redo_tooltips()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandChangeGroupStyle(QUndoCommand):
    """ меняет цвет группы и графиков группы"""

    def __init__(self, rs, style: dict, old_style: dict, idx: int, description: str) -> None:
        super(CommandChangeGroupStyle, self).__init__(description)
        self.setText(description)
        self.RS = rs
        self._idx = idx
        self._style = style
        self._old_style = old_style
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.GroupsTable = rs.ui.GroupsTable

    def redo(self) -> None:
        self.GroupsTable.model().change_cell_data(self._idx - 1, 1, self._style)
        self.RS.change_plot_color_for_group(self._idx, self._style)
        self.update_undo_redo_tooltips()

    def undo(self) -> None:
        self.GroupsTable.model().change_cell_data(self._idx - 1, 1, self._old_style)
        self.RS.change_plot_color_for_group(self._idx, self._old_style)
        self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.RS.set_modified()


class CommandUpdateInterpolated(QUndoCommand):
    """ for delete/undo_add rows in input_table, self.ImportedArray and input plot for interpolated files"""

    def __init__(self, rs, interpolated: list[tuple[str, np.ndarray]], description: str) -> None:
        super(CommandUpdateInterpolated, self).__init__(description)
        # print('CommandUpdateInterpolated')
        self.setText(description)
        self.Interpolated = interpolated  # 0 - name, 1 - array
        self.rs = rs
        self.ImportedArray_before = dict()
        self.interpolated_names = []
        for key, _ in self.Interpolated:
            self.interpolated_names.append(key)
            self.ImportedArray_before[key] = self.rs.ImportedArray[key]
        self.input_table = rs.ui.input_table
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop

    @asyncSlot()
    async def redo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() + (self.rs.time_start - datetime.now())
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)
        for key, array in self.Interpolated:
            self.rs.ImportedArray[key] = array
        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self._redo, i)
                       for i in self.interpolated_names]
        await gather(*futures)
        await self.rs.preprocessing.update_plot_item(self.rs.ImportedArray.items())
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() + (self.rs.time_start - datetime.now())
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)
        for key, _ in self.Interpolated:
            self.rs.ImportedArray[key] = self.ImportedArray_before[key]
        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self._undo, i)
                       for i in self.interpolated_names]
        await gather(*futures)
        await self.rs.preprocessing.update_plot_item(self.rs.ImportedArray.items())
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    def _redo(self, name: str) -> None:
        n_array = self.rs.ImportedArray[name]
        self.input_table.model().change_cell_data(name, 'Min, nm', n_array.min(axis=0)[0])
        self.input_table.model().change_cell_data(name, 'Max, nm', n_array.max(axis=0)[0])

    def _undo(self, name: str) -> None:
        n_array = self.ImportedArray_before[name]
        self.input_table.model().change_cell_data(name, 'Min, nm', n_array.min(axis=0)[0])
        self.input_table.model().change_cell_data(name, 'Max, nm', n_array.max(axis=0)[0])

    async def _stop(self) -> None:
        self.rs.disable_buttons(False)
        self.update_undo_redo_tooltips()
        self.rs.select_plots_by_buttons()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandUpdateDespike(QUndoCommand):
    """
    change array in self.ImportedArray
    update column 4 in input plot
    refresh input plot for despiked files,
    update self.BeforeDespike
    Parameters
    _____________________
    despiked_list: 0 - name 1 - despiked array, 2 - peaks nm
    """

    def __init__(self, rs, despiked_list: list[tuple[str, np.ndarray, list[float]]],
                 description: str) -> None:
        super(CommandUpdateDespike, self).__init__(description)
        # print('CommandUpdateDespike')
        self.rs = rs
        self.despiked_list = despiked_list
        self.setText(description)
        self.ImportedArray_before = dict()
        self.despiked_nm = dict()
        self.previous_text = dict()
        self.despiked_names = []
        self.input_table = rs.ui.input_table
        for i in despiked_list:
            self.ImportedArray_before[i[0]] = rs.ImportedArray[i[0]]
            self.despiked_nm[i[0]] = i[2]

        for i in self.despiked_list:
            name = i[0]
            self.despiked_names.append(name)
            self.previous_text[name] = self.input_table.model().cell_data(name, 'Despiked, nm')
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop

    @asyncSlot()
    async def redo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)

        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self._redo, i)
                       for i in self.despiked_list]
        await gather(*futures)
        await self.rs.preprocessing.update_plot_item(self.rs.ImportedArray.items())
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() + (self.rs.time_start - datetime.now())
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)

        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self._undo_dict, i)
                       for i in self.ImportedArray_before.items()]
        await gather(*futures)
        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self._undo_table, i)
                       for i in self.despiked_names]
        await gather(*futures)
        await self.rs.preprocessing.update_plot_item(self.rs.ImportedArray.items())
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    def _redo(self, i: tuple[str, np.ndarray, list[float]]) -> None:
        name = i[0]
        n_array = i[1]
        self.rs.ImportedArray[name] = n_array
        if name not in self.rs.preprocessing.BeforeDespike:
            self.rs.preprocessing.BeforeDespike[name] = self.ImportedArray_before[name]

        new_text = str(self.despiked_nm[name])
        new_text = new_text.replace('[', '')
        new_text = new_text.replace(']', '')
        previous_text = self.previous_text[name]
        if previous_text != '':
            new_text = previous_text + ', ' + new_text
        self.input_table.model().change_cell_data(name, 'Despiked, nm', new_text)

    def _undo_table(self, name: str) -> None:
        previous_text = self.previous_text[name]
        self.input_table.model().change_cell_data(name, 'Despiked, nm', previous_text)

    def _undo_dict(self, i: tuple[str, np.ndarray]) -> None:
        name = i[0]
        n_array = i[1]
        self.rs.ImportedArray[name] = n_array
        if name in self.rs.preprocessing.BeforeDespike:
            del self.rs.preprocessing.BeforeDespike[name]

    async def _stop(self) -> None:
        self.rs.disable_buttons(False)
        self.update_undo_redo_tooltips()
        self.rs.input_plot_widget_plot_item.getViewBox().updateAutoRange()
        self.rs.input_plot_widget_plot_item.updateParamList()
        self.rs.input_plot_widget_plot_item.recomputeAverages()
        self.rs.select_plots_by_buttons()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandConvert(QUndoCommand):
    """
    change array in self.ConvertedDict
    update column 5 Rayleigh line nm in input plot
    refresh converted plot for all files,
    """

    def __init__(self, rs, converted_list: list[tuple[str, np.ndarray, float, float]],
                 description: str) -> None:
        super(CommandConvert, self).__init__(description)
        self.rs = rs
        self.converted_list = converted_list  # 0 - name, 1 - array, 2 - Rayleigh line, 3- fwhm cm-1, 4 - SNR
        self.setText(description)
        self.ConvertedDict = rs.preprocessing.ConvertedDict
        self.ConvertedDictBefore = dict()
        self.ConvertedDictBefore = rs.preprocessing.ConvertedDict.copy()
        self.input_table = rs.ui.input_table
        self.change_list = []
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop
        for name, _, rayleigh_line, fwhm_cm, snr in self.converted_list:
            new_rl = rayleigh_line
            prev_rl = self.input_table.model().cell_data(name, 'Rayleigh line, nm')
            self.change_list.append((name, new_rl, prev_rl, fwhm_cm, snr))

    @asyncSlot()
    async def redo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)
        self.ConvertedDict.clear()
        for name, array, _, _, _ in self.converted_list:
            self.ConvertedDict[name] = array

        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self.change_column_rayleigh, i)
                       for i in self.change_list]
        await gather(*futures)

        await self.rs.preprocessing.update_plot_item(self.ConvertedDict.items(), 1)

        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)
        self.ConvertedDict.clear()
        self.ConvertedDict = self.ConvertedDictBefore.copy()

        with ThreadPoolExecutor() as executor:
            futures = [self.loop.run_in_executor(executor, self.change_column_rayleigh, i, True)
                       for i in self.change_list]
        await gather(*futures)
        await self.rs.preprocessing.update_plot_item(self.ConvertedDict.items(), 1)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        self.rs.disable_buttons(False)
        self.update_undo_redo_tooltips()
        self.rs.update_cm_min_max_range()
        self.rs.set_buttons_ability()
        self.rs.select_plots_by_buttons()
        self.rs.converted_cm_widget_plot_item.getViewBox().updateAutoRange()
        self.rs.converted_cm_widget_plot_item.updateParamList()
        self.rs.converted_cm_widget_plot_item.recomputeAverages()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)

    def change_column_rayleigh(self, i: tuple[str, float, float, float, float], undo: bool = False) -> None:
        name, new_rl, old_rl, fwhm_cm, snr = i
        if undo:
            self.input_table.model().change_cell_data(name, 'Rayleigh line, nm', old_rl)
        else:
            self.input_table.model().change_cell_data(name, 'Rayleigh line, nm', new_rl)
            self.input_table.model().change_cell_data(name, 'FWHM, cm\N{superscript minus}\N{superscript one}', fwhm_cm)
            self.input_table.model().change_cell_data(name, 'SNR', snr)


class CommandCutFirst(QUndoCommand):
    """
    change array in self.CuttedFirstDict
    refresh cut plot for all files,
    """

    def __init__(self, rs, cut_list: list[tuple[str, np.ndarray]], description: str) -> None:
        super(CommandCutFirst, self).__init__(description)
        self.rs = rs
        self.cut_list = cut_list  # 0 - name, 1 - array
        self.setText(description)
        self.CuttedFirstDict = rs.preprocessing.CuttedFirstDict
        self.CuttedFirstDictBefore = rs.preprocessing.CuttedFirstDict.copy()
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop

    @asyncSlot()
    async def redo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)
        self.CuttedFirstDict.clear()
        for name, array in self.cut_list:
            self.CuttedFirstDict[name] = array
        self.rs.cut_cm_plotItem.clear()
        await self.rs.preprocessing.update_plot_item(self.CuttedFirstDict.items(), 2)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)
        self.CuttedFirstDict.clear()
        self.CuttedFirstDict = self.CuttedFirstDictBefore.copy()
        self.rs.cut_cm_plotItem.clear()
        await self.rs.preprocessing.update_plot_item(self.CuttedFirstDict.items(), 2)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        self.rs.disable_buttons(False)
        self.update_undo_redo_tooltips()
        self.rs.set_buttons_ability()
        self.rs.select_plots_by_buttons()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandNormalize(QUndoCommand):

    def __init__(self, rs, normalized_list: list[tuple[str, np.ndarray]],
                 method: str, description: str) -> None:
        super(CommandNormalize, self).__init__(description)
        self.rs = rs
        self.normalized_list = normalized_list  # 0 - name, 1 - array
        self.setText(description)
        self.method = method
        self.method_old = rs.normalization_method
        self.NormalizedDictBefore = rs.preprocessing.NormalizedDict.copy()
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop

    @asyncSlot()
    async def redo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)
        self.rs.preprocessing.NormalizedDict.clear()
        for key, array in self.normalized_list:
            self.rs.preprocessing.NormalizedDict[key] = array
        await self.rs.preprocessing.update_plot_item(self.rs.preprocessing.NormalizedDict.items(), 3)
        self.rs.normalization_method = self.method
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)
        self.rs.preprocessing.NormalizedDict.clear()
        self.rs.preprocessing.NormalizedDict = self.NormalizedDictBefore.copy()
        info('Undo normalization. The len of NormalizedDict is %s', len(self.rs.preprocessing.NormalizedDict))
        await self.rs.preprocessing.update_plot_item(self.NormalizedDictBefore.items(), 3)
        self.rs.normalization_method = self.method_old
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        self.rs.disable_buttons(False)
        self.rs.ui.normalize_plot_widget.setTitle(
            "<span style=\"font-family: AbletonSans; color:" + self.rs.theme_colors[
                'plotText'] + ";font-size:14pt\">Normalized plots. Method: " + self.rs.normalization_method + "</span>")
        self.update_undo_redo_tooltips()
        self.rs.set_buttons_ability()
        self.rs.select_plots_by_buttons()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandSmooth(QUndoCommand):

    def __init__(self, rs, smoothed_list: list[tuple[str, np.ndarray]],
                 method: str, params: int | float, description: str) -> None:
        super(CommandSmooth, self).__init__(description)
        self.rs = rs
        self.smoothed_list = smoothed_list  # 0 - name, 1 - array
        self.setText(description)
        self.method = method
        self.params = params
        self.method_old = rs.smooth_method
        self.method_new = self.generate_title_text()
        self.smoothed_dataset_old = copy.deepcopy(rs.ui.smoothed_dataset_table_view.model().dataframe())
        self.smoothed_dataset_new = copy.deepcopy(self.create_smoothed_dataset_new())
        self.SmoothedDictBefore = rs.preprocessing.smoothed_spectra.copy()
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop

    def create_smoothed_dataset_new(self) -> pd.DataFrame:
        filename_group = self.rs.ui.input_table.model().column_data(2)
        x_axis = self.smoothed_list[0][1][:, 0]
        columns_params = []
        for i in x_axis:
            columns_params.append('k%s' % np.round(i, 2))
        df = pd.DataFrame(columns=columns_params)
        class_ids = []
        indexes = list(filename_group.index)
        for filename, n_array in self.smoothed_list:
            class_ids.append(filename_group.loc[filename])
            df2 = pd.DataFrame(n_array[:, 1].reshape(1, -1), columns=columns_params)
            df = pd.concat([df, df2], ignore_index=True)
        df2 = pd.DataFrame({'Class': class_ids, 'Filename': indexes})
        df = pd.concat([df2, df], axis=1)
        return df

    def generate_title_text(self) -> str:
        text = self.method + '. '
        match self.method:
            case 'EMD':
                text += 'IMFs: ' + str(self.params)
            case 'MLESG':
                text += 'sigma: ' + str(self.params[1])
            case 'EEMD' | 'CEEMDAN':
                imfs, trials = self.params
                text += 'IMFs: ' + str(imfs) + ', trials: ' + str(trials)
            case 'Savitsky-Golay filter':
                window_len, polyorder = self.params
                text += 'Window length: ' + str(window_len) + ', polynome order: ' + str(polyorder)
            case 'Whittaker smoother':
                text += 'λ: ' + str(self.params)
            case 'Flat window':
                text += 'Window length: ' + str(self.params)
            case 'hanning' | 'hamming' | 'bartlett' | 'blackman':
                text += 'Window length: ' + str(self.params)
            case 'kaiser':
                window_len, kaiser_beta = self.params
                text += 'Window length: ' + str(window_len) + ', β: ' + str(kaiser_beta)
            case 'Median filter':
                text += 'Window length: ' + str(self.params)
            case 'Wiener filter':
                text += 'Window length: ' + str(self.params)
        return text

    @asyncSlot()
    async def redo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)
        self.rs.preprocessing.smoothed_spectra.clear()
        for key, array in self.smoothed_list:
            self.rs.preprocessing.smoothed_spectra[key] = array
        await self.rs.preprocessing.update_plot_item(self.rs.preprocessing.smoothed_spectra.items(), 4)
        self.rs.smooth_method = self.method_new
        self.rs.ui.smoothed_dataset_table_view.model().set_dataframe(self.smoothed_dataset_new)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)
        self.rs.preprocessing.smoothed_spectra.clear()
        self.rs.preprocessing.smoothed_spectra = self.SmoothedDictBefore.copy()
        info('Undo smoothing. The len of SmoothedDict is %s', len(self.rs.preprocessing.smoothed_spectra))
        await self.rs.preprocessing.update_plot_item(self.SmoothedDictBefore.items(), 4)
        self.rs.smooth_method = self.method_old
        self.rs.ui.smoothed_dataset_table_view.model().set_dataframe(self.smoothed_dataset_old)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        self.rs.disable_buttons(False)
        self.rs.ui.smooth_plot_widget.setTitle(
            "<span style=\"font-family: AbletonSans; color:" + self.rs.theme_colors[
                'plotText'] + ";font-size:14pt\">Smoothed plots. Method: " + self.rs.smooth_method + "</span>")
        self.update_undo_redo_tooltips()
        self.rs.set_buttons_ability()
        self.rs.select_plots_by_buttons()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandBaselineCorrection(QUndoCommand):

    def __init__(self, rs, baseline_corrected_list: list[tuple[str, np.ndarray, np.ndarray]],
                 method: str, params: int | float | None, description: str) -> None:
        super(CommandBaselineCorrection, self).__init__(description)
        self.rs = rs
        self.baseline_corrected_list = baseline_corrected_list  # 0 - name, 1 - baseline, 2 y_new corrected
        self.setText(description)
        self.method = method
        self.params = params
        self.method_old = rs.baseline_method
        self.method_new = self.generate_title_text()
        self.baseline_corrected_dict_before = rs.preprocessing.baseline_corrected_dict.copy()
        self.baseline_dict_before = rs.preprocessing.baseline_dict.copy()
        self.dataset_old = copy.deepcopy(rs.ui.baselined_dataset_table_view.model().dataframe())
        self.dataset_new = copy.deepcopy(self.create_baseline_corrected_dataset_new())
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop

    def create_baseline_corrected_dataset_new(self) -> pd.DataFrame:
        filename_group = self.rs.ui.input_table.model().column_data(2)
        x_axis = self.baseline_corrected_list[0][1][:, 0]
        columns_params = []
        for i in x_axis:
            columns_params.append('k%s' % np.round(i, 2))
        df = pd.DataFrame(columns=columns_params)
        class_ids = []
        indexes = list(filename_group.index)
        for filename, _, n_array in self.baseline_corrected_list:
            class_ids.append(filename_group.loc[filename])
            df2 = pd.DataFrame(n_array[:, 1].reshape(1, -1), columns=columns_params)
            df = pd.concat([df, df2], ignore_index=True)
        df2 = pd.DataFrame({'Class': class_ids, 'Filename': indexes})
        df = pd.concat([df2, df], axis=1)
        return df

    def generate_title_text(self) -> str:
        text = self.method + '. '
        match self.method:
            case 'Poly':
                text += 'Polynome order: ' + str(self.params)
            case 'ModPoly' | 'iModPoly' | 'ExModPoly':
                text += 'Polynome order: ' + str(self.params[0]) + ', Δ: ' + str(self.params[1])
            case 'Penalized poly':
                text += 'Polynome order: ' + str(self.params[0]) + ', Δ: ' + str(self.params[1]) \
                        + ', α-factor: ' + str(self.params[3]) + ', cost-function: ' + str(self.params[4])
            case 'LOESS':
                text += 'Polynome order: ' + str(self.params[0]) + ', Δ: ' + str(self.params[1]) \
                        + ', fraction: ' + str(self.params[3]) + ', scale: ' + str(self.params[4])
            case 'Quantile regression':
                text += 'Polynome order: ' + str(self.params[0]) + ', Δ: ' + str(self.params[1]) \
                        + ', quantile: ' + str(self.params[3])
            case 'Goldindec':
                text += 'Polynome order: ' + str(self.params[0]) + ', Δ: ' + str(self.params[1]) \
                        + ', cost-function: ' + str(self.params[3]) + ', peak ratio: ' + str(self.params[4]) \
                        + ', α-factor: ' + str(self.params[5])
            case 'AsLS' | 'iAsLS' | 'psaLSA' | 'DerPSALSA' | 'MPLS' | 'arPLS' | 'airPLS' | 'iarPLS' | 'asPLS':
                text += 'λ: ' + str(self.params[0]) + ', p: ' + str(self.params[1])
            case 'drPLS':
                text += 'λ: ' + str(self.params[0]) + ', ratio: ' + str(self.params[1]) + ', η: ' + str(self.params[3])
            case 'iMor' | 'MorMol' | 'AMorMol' | 'JBCD':
                text += 'tol: ' + str(self.params[1])
            case 'MPSpline':
                text += 'λ: ' + str(self.params[0]) + ', p: ' + str(self.params[1]) \
                        + ', spline degree: ' + str(self.params[2])
            case 'Mixture Model':
                text += 'λ: ' + str(self.params[0]) + ', p: ' + str(self.params[1]) \
                        + ', spline degree: ' + str(self.params[2]) + ', tol: ' + str(self.params[4])
            case 'IRSQR':
                text += 'λ: ' + str(self.params[0]) + ', quantile: ' + str(self.params[1]) \
                        + ', spline degree: ' + str(self.params[2])
            case 'RIA':
                text += 'tol: ' + str(self.params)
            case 'Dietrich':
                text += 'num_std: ' + str(self.params[0]) + ', poly order: ' + str(self.params[1]) \
                        + ', tol: ' + str(self.params[2]) + ', interp half-window: ' + str(self.params[4]) \
                        + ', min_length: ' + str(self.params[5])
            case 'Golotvin':
                text += 'num_std: ' + str(self.params[0]) + ', interp half-window: ' + str(self.params[1]) \
                        + ', min_length: ' + str(self.params[2]) + ', sections: ' + str(self.params[3])
            case 'Std Distribution':
                text += 'num_std: ' + str(self.params[0]) + ', interp half-window: ' + str(self.params[1]) \
                        + ', fill half-window: ' + str(self.params[2])
            case 'FastChrom':
                text += 'interp half-window: ' + str(self.params[0]) + ', min_length: ' + str(self.params[2])
            case 'FABC':
                text += 'λ: ' + str(self.params[0]) + ', num_std: ' + str(self.params[1]) \
                        + ', min_length: ' + str(self.params[2])
            case 'OER' | 'Adaptive MinMax':
                text += 'Optimized method: ' + self.params
        return text

    @asyncSlot()
    async def redo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)
        self.rs.preprocessing.baseline_corrected_dict.clear()
        self.rs.preprocessing.baseline_dict.clear()
        for key, baseline, array_corrected in self.baseline_corrected_list:
            self.rs.preprocessing.baseline_corrected_dict[key] = array_corrected
            self.rs.preprocessing.baseline_corrected_not_trimmed_dict[key] = array_corrected
            self.rs.preprocessing.baseline_dict[key] = baseline
        await self.rs.preprocessing.update_plot_item(self.rs.preprocessing.baseline_corrected_dict.items(), 5)
        self.rs.baseline_method = self.method_new
        self.rs.ui.baselined_dataset_table_view.model().set_dataframe(self.dataset_new)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)
        self.rs.preprocessing.baseline_corrected_dict.clear()
        self.rs.preprocessing.baseline_dict.clear()
        self.rs.preprocessing.baseline_corrected_dict = self.baseline_corrected_dict_before.copy()
        self.rs.preprocessing.baseline_corrected_not_trimmed_dict = self.baseline_corrected_dict_before.copy()
        self.rs.preprocessing.baseline_dict = self.baseline_dict_before.copy()
        await self.rs.preprocessing.update_plot_item(self.baseline_corrected_dict_before.items(), 5)
        self.rs.baseline_method = self.method_old
        self.rs.ui.baselined_dataset_table_view.model().set_dataframe(self.dataset_old)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        self.rs.disable_buttons(False)
        self.rs.ui.baseline_plot_widget.setTitle(
            "<span style=\"font-family: AbletonSans; color:" + self.rs.theme_colors[
                'plotText'] + ";font-size:14pt\">Baseline corrected plots. Method: " +
            self.rs.baseline_method + "</span>")
        self.update_undo_redo_tooltips()
        self.rs.set_buttons_ability()
        self.rs.select_plots_by_buttons()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandTrim(QUndoCommand):
    """
    change array in self.baseline_corrected_dict
    refresh baseline plot for all files,
    """

    def __init__(self, rs, cut_list: list[tuple[str, np.ndarray]], description: str) -> None:
        super(CommandTrim, self).__init__(description)
        self.rs = rs
        self.cut_list = cut_list  # 0 - name, 1 - array
        self.setText(description)
        self.baseline_corrected_dict = rs.preprocessing.baseline_corrected_dict
        self.baseline_corrected_dict_before = rs.preprocessing.baseline_corrected_dict.copy()
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop

    @asyncSlot()
    async def redo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        self.rs.disable_buttons(True)
        self.baseline_corrected_dict.clear()
        for name, array in self.cut_list:
            self.baseline_corrected_dict[name] = array
        self.rs.baseline_corrected_plotItem.clear()
        await self.rs.preprocessing.update_plot_item(self.baseline_corrected_dict.items(), 5)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        self.rs.disable_buttons(True)
        self.baseline_corrected_dict.clear()
        self.rs.preprocessing.baseline_corrected_dict = self.baseline_corrected_dict_before.copy()
        self.rs.baseline_corrected_plotItem.clear()
        await self.rs.preprocessing.update_plot_item(self.rs.preprocessing.baseline_corrected_dict.items(), 5)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        self.rs.disable_buttons(False)
        self.update_undo_redo_tooltips()
        self.rs.set_buttons_ability()
        self.rs.select_plots_by_buttons()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Action completed for ' + str(seconds) + ' sec.', 25000)
        # print('CommandTrim')
        self.rs.set_modified()
        collect(2)
        await self.rs.preprocessing.update_averaged()
        await sleep(0)


class CommandAddDeconvLine(QUndoCommand):
    """
    add row to self.ui.deconv_lines_table
    add curve to deconvolution_plotItem
    """

    def __init__(self, rs, idx: int, line_type: str, description: str) -> None:
        super(CommandAddDeconvLine, self).__init__(description)
        self.RS = rs
        self.setText(description)
        self._idx = idx
        self._legend = 'Curve ' + str(idx + 1)
        self._line_type = line_type
        self._line_params = rs.fitting.initial_peak_parameters(line_type)
        self._style = random_line_style()
        self._line_param_names = self._line_params.keys()
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop

    @asyncSlot()
    async def redo(self) -> None:
        info('CommandAddDeconvLine redo, line index %s', self._idx)
        self.RS.ui.deconv_lines_table.model().append_row(self._legend, self._line_type, self._style, self._idx)
        for param_name in self._line_param_names:
            if param_name != 'x_axis':
                self.RS.ui.fit_params_table.model().append_row(self._idx, param_name,
                                                               self._line_params[param_name])
        self.RS.fitting.add_deconv_curve_to_plot(self._line_params, self._idx, self._style, self._line_type)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        info('CommandAddDeconvLine undo, line index %s', self._idx)
        self.RS.ui.deconv_lines_table.model().delete_row(self._idx)
        self.RS.fitting.delete_deconv_curve(self._idx)
        self.RS.ui.fit_params_table.model().delete_rows(self._idx)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        self.RS.fitting.draw_sum_curve()
        self.RS.fitting.draw_residual_curve()
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.RS.set_buttons_ability()

    async def _stop(self) -> None:
        # print('CommandAddDeconvLine')
        self.update_undo_redo_tooltips()
        self.RS.set_modified()
        collect(2)
        await sleep(0)


class CommandDeleteDeconvLines(QUndoCommand):
    """ delete row in deconv_lines_table, remove curve from deconvolution_plotItem, remove rows from parameters table"""

    def __init__(self, rs, description: str) -> None:
        super(CommandDeleteDeconvLines, self).__init__(description)
        self.setText(description)
        self.deconv_lines_table = rs.ui.deconv_lines_table
        selected_row = self.deconv_lines_table.selectionModel().selectedIndexes()[0].row()
        self._selected_row = int(self.deconv_lines_table.model().row_data(selected_row).name)
        fit_lines_df = self.deconv_lines_table.model().query_result('index == %s' % self._selected_row)
        self._legend = fit_lines_df['Legend'][self._selected_row]
        self._line_type = fit_lines_df['Type'][self._selected_row]
        self._style = fit_lines_df['Style'][self._selected_row]
        self._line_params = rs.fitting.current_line_parameters(self._selected_row)
        self._line_params['x_axis'] = rs.fitting.x_axis_for_line(self._line_params['x0'], self._line_params['dx'])
        self._df = rs.ui.fit_params_table.model().dataframe().query(
            'line_index == %s' % self._selected_row)
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.RS = rs
        self.loop = rs.loop

    @asyncSlot()
    async def redo(self) -> None:
        self.RS.ui.deconv_lines_table.model().delete_row(self._selected_row)
        self.RS.fitting.delete_deconv_curve(self._selected_row)
        self.RS.ui.fit_params_table.model().delete_rows(self._selected_row)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        self.RS.ui.deconv_lines_table.model().append_row(self._legend, self._line_type, self._style, self._selected_row)
        self.RS.fitting.add_deconv_curve_to_plot(self._line_params, self._selected_row, self._style, self._line_type)
        self.RS.ui.fit_params_table.model().concat_df(self._df)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.RS.set_buttons_ability()

    async def _stop(self) -> None:
        # print('CommandDeleteDeconvLines')
        self.RS.fitting.draw_sum_curve()
        self.RS.fitting.draw_residual_curve()
        self.RS.fitting.deselect_selected_line()
        self.update_undo_redo_tooltips()
        self.RS.set_modified()
        collect(2)
        await sleep(0)


class CommandDeconvLineTypeChanged(QUndoCommand):
    """
    add row to self.ui.deconv_lines_table
    add curve to deconvolution_plotItem
    """

    def __init__(self, rs, line_type_new: str, line_type_old: str, idx: int, description: str) \
            -> None:
        super(CommandDeconvLineTypeChanged, self).__init__(description)
        self.RS = rs
        self._idx = idx
        self.line_type_new = line_type_new
        self.line_type_old = line_type_old
        self.setText(description)
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack

    @asyncSlot()
    async def redo(self) -> None:
        info('CommandDeconvLineTypeChanged redo, line index %s', self._idx)
        self.RS.ui.deconv_lines_table.model().set_cell_data_by_idx_col_name(self._idx, 'Type', self.line_type_new)
        self.update_params_table(self.line_type_new, self.line_type_old)
        self.RS.fitting.redraw_curve(line_type=self.line_type_new, idx=self._idx)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        info('CommandDeconvLineTypeChanged undo, line index %s', self._idx)
        self.RS.ui.deconv_lines_table.model().set_cell_data_by_idx_col_name(self._idx, 'Type', self.line_type_old)
        self.update_params_table(self.line_type_old, self.line_type_new)
        self.RS.fitting.redraw_curve(line_type=self.line_type_old, idx=self._idx)
        create_task(self._stop())

    def update_params_table(self, line_type_new: str, line_type_old: str) -> None:
        params_old = self.RS.peak_shapes_params[line_type_old]['add_params'] \
            if 'add_params' in self.RS.peak_shapes_params[line_type_old] else None
        params_new = self.RS.peak_shapes_params[line_type_new]['add_params'] \
            if 'add_params' in self.RS.peak_shapes_params[line_type_new] else None
        if params_old == params_new or (params_old is None and params_new is None):
            return
        params_to_add = []
        params_to_dlt = []
        if params_old is None:
            params_to_add = params_new
        elif params_new is not None:
            for i in params_new:
                if i not in params_old:
                    params_to_add.append(i)
        if params_new is None:
            params_to_dlt = params_old
        elif params_old is not None:
            for i in params_old:
                if i not in params_new:
                    params_to_dlt.append(i)
        line_params = self.RS.fitting.initial_peak_parameters(line_type_new)
        for i in params_to_add:
            self.RS.ui.fit_params_table.model().append_row(self._idx, i, line_params[i])
        for i in params_to_dlt:
            self.RS.ui.fit_params_table.model().delete_rows_multiindex(('', self._idx, i))

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.RS.set_buttons_ability()

    async def _stop(self) -> None:
        # print('CommandDeconvLineTypeChanged')
        self.RS.fitting.draw_sum_curve()
        self.RS.fitting.draw_residual_curve()
        self.update_undo_redo_tooltips()
        self.RS.set_modified()
        collect(2)
        await sleep(0)


class CommandUpdateDeconvCurveStyle(QUndoCommand):

    def __init__(self, main_window, idx: int, style: dict, old_style: dict, description: str) -> None:
        super(CommandUpdateDeconvCurveStyle, self).__init__(description)
        self.setText(description)
        self._style = style
        self._old_style = old_style
        self._idx = idx
        self.UndoAction = main_window.action_undo
        self.RedoAction = main_window.action_redo
        self.UndoStack = main_window.undoStack
        self.mw = main_window

    @asyncSlot()
    async def redo(self) -> None:
        self.mw.fitting.update_curve_style(self._idx, self._style)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        self.mw.fitting.update_curve_style(self._idx, self._old_style)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        self.update_undo_redo_tooltips()
        self.mw.set_modified()
        collect(2)
        await sleep(0)


class CommandUpdateDataCurveStyle(QUndoCommand):

    def __init__(self, rs, style: dict, old_style: dict, curve_type: str, description: str) -> None:
        super(CommandUpdateDataCurveStyle, self).__init__(description)
        # print('CommandUpdateDataCurveStyle')
        self.setText(description)
        self._style = style
        self._old_style = old_style
        self._curve_type = curve_type
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.RS = rs

    @asyncSlot()
    async def redo(self) -> None:
        self.set_pen(self._style)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        self.set_pen(self.old_style)
        create_task(self._stop())

    def set_pen(self, style) -> None:
        color = style['color']
        color.setAlphaF(1.0)
        pen = mkPen(color=color, style=style['style'], width=style['width'])
        if self._curve_type == 'data':
            self.RS.fitting.data_curve.setPen(pen)
            self.RS.data_style_button_style_sheet(color.name())
            self.RS.fitting.data_style = style
        elif self._curve_type == 'sum':
            self.RS.fitting.sum_curve.setPen(pen)
            self.RS.sum_style_button_style_sheet(color.name())
            self.RS.fitting.sum_style = style
        elif self._curve_type == 'residual':
            self.RS.fitting.residual_curve.setPen(pen)
            self.RS.residual_style_button_style_sheet(color.name())
            self.RS.fitting.residual_style = style
        elif self._curve_type == 'sigma3':
            pen, brush = curve_pen_brush_by_style(style)
            self.RS.fitting.fill.setPen(pen)
            self.RS.fitting.fill.setBrush(brush)
            self.RS.sigma3_style_button_style_sheet(color.name())
            self.RS.fitting.sigma3_style = style

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        # print('CommandUpdateDataCurveStyle')
        self.update_undo_redo_tooltips()
        self.RS.set_modified()
        collect(2)
        await sleep(0)


class CommandUpdateTableCell(QUndoCommand):
    """Using in PandasModelGroupsTable and PandasModelDeconvLinesTable"""

    def __init__(self, obj, rs, index: QModelIndex, value: str,
                 old_value: str, description: str) -> None:
        super(CommandUpdateTableCell, self).__init__(description)
        # print('init CommandUpdateTableCell')
        self.setText(description)
        self._obj = obj
        self._index = index
        self._value = value
        self._old_value = old_value
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack

    @asyncSlot()
    async def redo(self) -> None:
        self._obj.set_cell_data_by_index(self._index, self._value)
        self._obj.dataChanged.emit(self._index, self._index)
        self.update_undo_redo_tooltips()

    @asyncSlot()
    async def undo(self) -> None:
        self._obj.set_cell_data_by_index(self._index, self._old_value)
        self._obj.dataChanged.emit(self._index, self._index)
        self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        # print('CommandUpdateTableCell')
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')


class CommandDeconvLineDragged(QUndoCommand):
    """ UNDO/REDO change position of line ROI"""

    def __init__(self, rs, params: tuple[float, float, float],
                 old_params: tuple[float, float, float], roi: ROI, description: str) -> None:
        super(CommandDeconvLineDragged, self).__init__(description)
        # print('init CommandDeconvLineDragged')
        self.setText(description)
        self._params = params
        self._old_params = old_params
        self._roi = roi
        self.RS = rs
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack

    @asyncSlot()
    async def redo(self) -> None:
        set_roi_size_pos(self._params, self._roi)
        self.update_undo_redo_tooltips()

    @asyncSlot()
    async def undo(self) -> None:
        set_roi_size_pos(self._old_params, self._roi)
        self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        # print('CommandDeconvLineDragged')
        self.RS.fitting.draw_sum_curve()
        self.RS.fitting.draw_residual_curve()
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')


class CommandDeconvLineParameterChanged(QUndoCommand):

    def __init__(self, delegate: QStyledItemDelegate, rs, index: QModelIndex, new_value: float,
                 old_value: float,
                 model, line_index: int, param_name: str, description: str) -> None:
        super(CommandDeconvLineParameterChanged, self).__init__(description)
        # print('init CommandDeconvLineParameterChanged')
        self.setText(description)
        self.delegate = delegate
        self._index = index
        self._new_value = new_value
        self._old_value = old_value
        self._model = model
        self._line_index = line_index
        self._param_name = param_name
        self.RS = rs
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack

    @asyncSlot()
    async def redo(self) -> bool:
        self._model.setData(self._index, self._new_value, Qt.EditRole)
        self.delegate.sigLineParamChanged.emit(self._new_value, self._line_index, self._param_name)
        self.update_undo_redo_tooltips()

    @asyncSlot()
    async def undo(self) -> None:
        self._model.setData(self._index, self._old_value, Qt.EditRole)
        self.delegate.sigLineParamChanged.emit(self._old_value, self._line_index, self._param_name)
        self.update_undo_redo_tooltips()

    def update_undo_redo_tooltips(self) -> None:
        # print('CommandDeconvLineParameterChanged')
        self.RS.fitting.draw_sum_curve()
        self.RS.fitting.draw_residual_curve()
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')


class CommandClearAllDeconvLines(QUndoCommand):
    """ delete row in deconv_lines_table, remove curve from deconvolution_plotItem, remove rows from parameters table"""

    def __init__(self, rs, description: str) -> None:
        super(CommandClearAllDeconvLines, self).__init__(description)
        self.setText(description)
        self._deconv_lines_table_df = rs.ui.deconv_lines_table.model().dataframe()
        self._checked = rs.ui.deconv_lines_table.model().checked()
        self._deconv_params_table_df = rs.ui.fit_params_table.model().dataframe()
        self.report_result_old = rs.fitting.report_result.copy()
        self.sigma3_dict = copy.deepcopy(rs.fitting.sigma3)
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.rs = rs

    @asyncSlot()
    async def redo(self) -> None:
        self.rs.ui.deconv_lines_table.model().clear_dataframe()
        self.rs.ui.fit_params_table.model().clear_dataframe()
        self.rs.fitting.remove_all_lines_from_plot()
        self.rs.fitting.report_result.clear()
        self.rs.ui.report_text_edit.setText('')
        self.rs.fitting.sigma3.clear()
        self.rs.fitting.sigma3_top.setData(x=np.array([0]), y=np.array([0]))
        self.rs.fitting.sigma3_bottom.setData(x=np.array([0]), y=np.array([0]))
        self.rs.fitting.update_sigma3_curves('')
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        self.rs.ui.deconv_lines_table.model().append_dataframe(self._deconv_lines_table_df)
        self.rs.ui.deconv_lines_table.model().set_checked(self._checked)
        self.rs.ui.fit_params_table.model().append_dataframe(self._deconv_params_table_df)
        self.rs.fitting.report_result = self.report_result_old
        self.rs.fitting.show_current_report_result()
        self.rs.fitting.sigma3 = self.sigma3_dict
        filename = '' if self.rs.fitting.is_template else self.rs.fitting.current_spectrum_deconvolution_name
        self.rs.fitting.update_sigma3_curves(filename)
        await self.rs.fitting.draw_all_curves()
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        self.rs.fitting.draw_sum_curve()
        self.rs.fitting.draw_residual_curve()
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.rs.set_buttons_ability()

    async def _stop(self) -> None:
        # print('CommandClearAllDeconvLines')
        self.rs.fitting.draw_sum_curve()
        try:
            self.rs.fitting.draw_residual_curve()
        except:
            pass
        self.update_undo_redo_tooltips()
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandStartIntervalChanged(QUndoCommand):
    """
    undo / redo setValue of self.ui.interval_start_dsb
    CommandStartIntervalChanged(RS, new_value: float, description: str)

    Parameters
    ----------
    rs
        Main window class
    new_value : float
        New value of self.ui.interval_start_dsb
    old_value : float
        Old value from self.old_start_interval_value
    description : str
        Description to set in tooltip
    """

    def __init__(self, rs, new_value: float, old_value: float, description: str) -> None:
        super(CommandStartIntervalChanged, self).__init__(description)
        info('init CommandStartIntervalChanged {!s}, new - {!s}, old - {!s}'.format(str(description),
                                                                                    str(new_value),
                                                                                    str(old_value)))
        self.setText(description)
        self.new_value = new_value
        self.old_value = old_value
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.RS = rs

    @asyncSlot()
    async def redo(self) -> None:
        info('redo CommandStartIntervalChanged, value %s' % self.new_value)
        self.set_value(self.new_value)

    @asyncSlot()
    async def undo(self) -> None:
        info('undo CommandStartIntervalChanged, value %s' % self.old_value)
        self.set_value(self.old_value)

    def set_value(self, value: float) -> None:
        self.RS.old_start_interval_value = value
        if self.RS.ui.interval_start_dsb.value() == value:
            return
        self.RS.CommandStartIntervalChanged_allowed = False
        self.RS.ui.interval_start_dsb.setValue(value)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.RS.set_buttons_ability()

    async def _stop(self) -> None:
        info('stop CommandStartIntervalChanged')
        self.RS.CommandStartIntervalChanged_allowed = True
        self.update_undo_redo_tooltips()
        # print('CommandStartIntervalChanged')
        self.RS.set_modified()
        collect(2)
        await sleep(0)


class CommandEndIntervalChanged(QUndoCommand):
    """
    undo / redo setValue of self.ui.interval_end_dsb"
    CommandEndIntervalChanged(RS, new_value: float, description: str)

    Parameters
    ----------
    rs
        Main window class
    new_value : float
        New value of self.ui.interval_end_dsb
    old_value : float
        Old value from self.old_end_interval_value
    description : str
        Description to set in tooltip
    """

    def __init__(self, rs, new_value: float, old_value: float, description: str) -> None:
        super(CommandEndIntervalChanged, self).__init__(description)
        info('init CommandEndIntervalChanged {!s}, new - {!s}, old - {!s}'.format(str(description),
                                                                                  str(new_value),
                                                                                  str(old_value)))
        self.setText(description)
        self.new_value = new_value
        self.old_value = old_value
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.RS = rs

    @asyncSlot()
    async def redo(self) -> None:
        info('redo CommandEndIntervalChanged, value %s' % self.new_value)
        self.set_value(self.new_value)

    @asyncSlot()
    async def undo(self) -> None:
        info('undo CommandEndIntervalChanged, value %s' % self.old_value)
        self.set_value(self.old_value)

    def set_value(self, value: float) -> None:
        self.RS.old_end_interval_value = value
        if self.RS.ui.interval_end_dsb.value() == value:
            return
        self.RS.CommandEndIntervalChanged_allowed = False
        self.RS.ui.interval_end_dsb.setValue(value)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.RS.set_buttons_ability()

    async def _stop(self) -> None:
        info('stop CommandEndIntervalChanged')
        self.RS.CommandEndIntervalChanged_allowed = True
        self.update_undo_redo_tooltips()
        # print('CommandEndIntervalChanged')
        self.RS.set_modified()
        collect(2)
        await sleep(0)


class CommandAfterFitting(QUndoCommand):
    """
    1. Set parameters value
    2. Update graph
    3. Show report

    Parameters
    ----------
    rs
        Main window class
    results : list[ModelResult]
    idx_type_param_count_legend_func : list[tuple[int, str, int, str, callable]]
    filename : str
        self.current_spectrum_deconvolution_name - current spectrum
    description : str
        Description to set in tooltip
    """

    def __init__(self, rs, results: list[ModelResult],
                 idx_type_param_count_legend_func: list[tuple[int, str, int, str, callable]], filename: str,
                 description: str) -> None:
        super(CommandAfterFitting, self).__init__(description)
        self.sigma3_top = np.array([])
        self.sigma3_bottom = np.array([])
        self.rs = rs
        self.report_text = ''
        self.results = results
        self.idx_type_param_count_legend_func = idx_type_param_count_legend_func.copy()
        self.filename = filename
        self.df = rs.ui.fit_params_table.model().query_result('filename == %r' % filename)
        self.sum_ar = rs.fitting.sum_array()
        self.fit_report = ''
        self.report_result_old = rs.fitting.report_result[filename] if filename in rs.fitting.report_result else ''
        self.sigma3 = self.rs.fitting.sigma3[self.filename] if self.filename in self.rs.fitting.sigma3 else None
        self.setText(description)
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop
        self.prepare_data()

    def prepare_data(self) -> None:
        av_text, sigma3_top, sigma3_bottom = fitting_metrics(self.results)
        for fit_result in self.results:
            self.fit_report += self.edited_fit_report(fit_result.fit_report(show_correl=False),
                                                      fit_result) + '\n' + '\n'
        x_axis, _ = self.sum_ar
        if self.sigma3_top.shape[0] < x_axis.shape[0]:
            d = x_axis.shape[0] - self.sigma3_top.shape[0]
            zer = np.zeros(d)
            self.sigma3_top = np.concatenate((self.sigma3_top, zer))
        if self.sigma3_bottom.shape[0] < x_axis.shape[0]:
            d = x_axis.shape[0] - self.sigma3_bottom.shape[0]
            zer = np.zeros(d)
            self.sigma3_bottom = np.concatenate((self.sigma3_bottom, zer))

        if av_text:
            self.report_text = av_text + self.fit_report + '\n' + '\n'
        else:
            self.report_text = self.fit_report

    @asyncSlot()
    async def redo(self) -> None:
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        info('redo CommandAfterFitting, filename %s' % self.filename)
        if self.filename != '':
            self.rs.ui.fit_params_table.model().delete_rows_by_filenames([self.filename])
            self.rs.fitting.add_line_params_from_template(self.filename)
        for fit_result in self.results:
            self.rs.fitting.set_parameters_after_fit_for_spectrum(fit_result, self.filename)
        x_axis, _ = self.sum_ar
        self.rs.fitting.sigma3[self.filename] = x_axis, self.sigma3_top, self.sigma3_bottom
        self.rs.fitting.update_sigma3_curves(self.filename)
        self.rs.fitting.report_result[self.filename] = self.report_text
        self.rs.ui.report_text_edit.setText(self.report_text)
        create_task(self._stop())

    def edited_fit_report(self, fit_report: str, res: lmfit.model.ModelResult) -> str:
        param_legend = []
        line_types = self.rs.ui.deconv_lines_table.model().get_visible_line_types()
        for key in res.best_values.keys():
            idx_param_name = key.split('_', 2)
            idx = int(idx_param_name[1])
            param_name = idx_param_name[2]
            legend = line_types.loc[idx].Legend
            param_legend.append((key, legend + ' ' + param_name))
        for old, new in param_legend:
            fit_report = fit_report.replace(old, new)
        if '[[Fit Statistics]]' in fit_report:
            idx = fit_report.find('[[Fit Statistics]]')
            fit_report = fit_report[idx:]
        return fit_report

    @asyncSlot()
    async def undo(self) -> None:
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        info('Undo CommandAfterFitting, filename %s' % self.filename)
        self.rs.ui.fit_params_table.model().delete_rows_by_filenames([self.filename])
        self.rs.ui.fit_params_table.model().concat_df(self.df)
        self.rs.fitting.set_rows_visibility()
        x_axis, y_axis = self.sum_ar
        self.rs.fitting.sum_curve.setData(x=x_axis, y=y_axis)
        self.rs.fitting.report_result[self.filename] = self.report_result_old
        self.rs.fitting.show_current_report_result()
        if self.sigma3 is not None:
            self.rs.fitting.sigma3[self.filename] = self.sigma3[0], self.sigma3[1], self.sigma3[2]
        else:
            del self.rs.fitting.sigma3[self.filename]
            self.rs.fitting.fill.setVisible(False)
        self.rs.fitting.update_sigma3_curves(self.filename)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        # print('stop CommandAfterFitting')
        info('stop CommandAfterFitting')
        self.rs.fitting.redraw_curves_for_filename()
        self.rs.fitting.draw_sum_curve()
        self.rs.fitting.draw_residual_curve()
        self.rs.fitting.show_current_report_result()
        self.rs.fitting.update_sigma3_curves()
        self.rs.fitting.set_rows_visibility()
        self.rs.ui.fit_params_table.model().sort_index()
        self.rs.ui.fit_params_table.model().model_reset_emit()
        self.update_undo_redo_tooltips()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Fitting completed for ' + str(seconds) + ' sec.', 25000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandAfterBatchFitting(QUndoCommand):
    """
    1. Set parameters value
    2. Update graph
    3. Update / Show report

    Parameters
    ----------
    rs
        Main window class
    results : list[str, ModelResult]: filename - model result
    idx_type_param_count_legend_func : list[tuple[int, str, int, str, callable]]
    description : str
        Description to set in tooltip
    """

    def __init__(self, rs, results: list[tuple[str, ModelResult]],
                 idx_type_param_count_legend_func: list[tuple[int, str, int, str, callable]],
                 dely: list[tuple[str, np.ndarray]], description: str) -> None:
        super(CommandAfterBatchFitting, self).__init__(description)
        self.av_text = ''
        self.dely = dely
        self.rs = rs
        self.results = results
        self.idx_type_param_count_legend_func = copy.deepcopy(idx_type_param_count_legend_func)
        self.df_fit_params = copy.deepcopy(rs.ui.fit_params_table.model().dataframe())
        self.dataset_old = copy.deepcopy(rs.ui.deconvoluted_dataset_table_view.model().dataframe())
        self.dataset_new = None
        self.sum_ar = rs.fitting.sum_array()
        self.fit_report = ''
        self.report_result_old = copy.deepcopy(rs.fitting.report_result)
        self.sigma3 = copy.deepcopy(rs.fitting.sigma3)
        self.setText(description)
        self.fit_reports = {}
        self.chisqr_av = {}
        self.redchi_av = {}
        self.aic_av = {}
        self.bic_av = {}
        self.rsquared_av = {}
        self.keys = []
        self.sigma3_conc_up = {}
        self.sigma3_conc_bottom = {}
        self.report_text = {}
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.loop = rs.loop
        self.prepare_data()

    @asyncSlot()
    async def prepare_data(self) -> None:
        for key, _ in self.results:
            if key not in self.keys:
                self.keys.append(key)
        for key in self.keys:
            self.fit_reports[key] = ''
            self.chisqr_av[key] = []
            self.redchi_av[key] = []
            self.aic_av[key] = []
            self.bic_av[key] = []
            self.rsquared_av[key] = []
            self.sigma3_conc_up[key] = np.array([])
            self.sigma3_conc_bottom[key] = np.array([])
        line_types = self.rs.ui.deconv_lines_table.model().get_visible_line_types()
        x_axis, _ = self.sum_ar
        for key, fit_result in self.results:
            self.fit_reports[key] += self.edited_fit_report(fit_result, line_types) + '\n'
            if fit_result.chisqr != 1e-250:
                self.chisqr_av[key].append(fit_result.chisqr)
            self.redchi_av[key].append(fit_result.redchi)
            self.aic_av[key].append(fit_result.aic)
            if fit_result.bic != -np.inf:
                self.bic_av[key].append(fit_result.bic)
            try:
                self.rsquared_av[key].append(fit_result.rsquared)
            except:
                # print("fit_result.rsquared error")
                debug("fit_result.rsquared error")
        for i, item in enumerate(self.results):
            key, fit_result = item
            self.sigma3_conc_up[key] = np.concatenate((self.sigma3_conc_up[key], fit_result.best_fit + self.dely[i][1]))
            self.sigma3_conc_bottom[key] = np.concatenate((self.sigma3_conc_bottom[key],
                                                           fit_result.best_fit - self.dely[i][1]))
        # check that shape of sigma curves = shape of mutual x_axis
        for key in self.keys:
            if self.sigma3_conc_up[key].shape[0] < x_axis.shape[0]:
                d = x_axis.shape[0] - self.sigma3_conc_up[key].shape[0]
                zer = np.zeros(d)
                self.sigma3_conc_up[key] = np.concatenate((self.sigma3_conc_up[key], zer))
            if self.sigma3_conc_bottom[key].shape[0] < x_axis.shape[0]:
                d = x_axis.shape[0] - self.sigma3_conc_bottom[key].shape[0]
                zer = np.zeros(d)
                self.sigma3_conc_bottom[key] = np.concatenate((self.sigma3_conc_bottom[key], zer))
        ranges = int(len(self.results) / len(self.bic_av))
        for key in self.keys:
            if ranges > 1:
                chisqr_av = np.round(np.mean(self.chisqr_av[key]), 6)
                redchi_av = np.round(np.mean(self.redchi_av[key]), 6)
                aic_av = np.round(np.mean(self.aic_av[key]), 6)
                bic_av = np.round(np.mean(self.bic_av[key]), 6)
                rsquared_av = np.round(np.mean(self.rsquared_av[key]), 8)
                av_text = "[[Average For Spectrum Fit Statistics]]" + '\n' \
                          + f"    chi-square         = {chisqr_av}" + '\n' \
                          + f"    reduced chi-square = {redchi_av}" + '\n' \
                          + f"    Akaike info crit   = {aic_av}" + '\n' \
                          + f"    Bayesian info crit = {bic_av}" + '\n' \
                          + f"    R-squared          = {rsquared_av}" + '\n' + '\n'
                self.report_text[key] = av_text + self.fit_reports[key]
            else:
                self.report_text[key] = self.fit_reports[key]
        l_chisqr_av = list(self.chisqr_av.values())
        flat_list = [item for sublist in l_chisqr_av for item in sublist]
        chisqr_av = np.round(np.mean(flat_list), 6)
        l_redchi_av = list(self.redchi_av.values())
        flat_list = [item for sublist in l_redchi_av for item in sublist]
        redchi_av = np.round(np.mean(flat_list), 6)
        l_aic_av = list(self.aic_av.values())
        flat_list = [item for sublist in l_aic_av for item in sublist]
        aic_av = np.round(np.mean(flat_list), 6)
        l_bic_av = list(self.bic_av.values())
        flat_list = [item for sublist in l_bic_av for item in sublist]
        bic_av = np.round(np.mean(flat_list), 6)
        l_rsquared_av = list(self.rsquared_av.values())
        flat_list = [item for sublist in l_rsquared_av for item in sublist]
        rsquared_av = np.round(np.mean(flat_list), 8)
        self.av_text = "[[Усредненная статистика по всем спектрам]]" + '\n' \
                       + f"    chi-square         = {chisqr_av}" + '\n' \
                       + f"    reduced chi-square = {redchi_av}" + '\n' \
                       + f"    Akaike info crit   = {aic_av}" + '\n' \
                       + f"    Bayesian info crit = {bic_av}" + '\n' \
                       + f"    R-squared          = {rsquared_av}" + '\n' + '\n'

    @asyncSlot()
    async def redo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Redo...' + self.text())
        info('redo CommandAfterBatchFitting')
        # print('redo CommandAfterBatchFitting')
        self.rs.fitting.report_result.clear()
        self.rs.ui.fit_params_table.model().delete_rows_by_filenames(self.keys)
        self.rs.fitting.add_line_params_from_template_batch(self.keys)
        for key, fit_result in self.results:
            self.rs.fitting.set_parameters_after_fit_for_spectrum(fit_result, key)
        x_axis, _ = self.sum_ar
        for key in self.keys:
            self.rs.fitting.sigma3[key] = x_axis, self.sigma3_conc_up[key], self.sigma3_conc_bottom[key]
        for key in self.keys:
            self.rs.fitting.report_result[key] = self.report_text[key]
        self.dataset_new = copy.deepcopy(self.rs.fitting.create_deconvoluted_dataset_new())
        self.rs.ui.deconvoluted_dataset_table_view.model().set_dataframe(self.dataset_new)
        for i in self.rs.fitting.report_result:
            self.rs.fitting.report_result[i] += '\n' + '\n' + self.av_text

        create_task(self._stop())

    @staticmethod
    def edited_fit_report(fit_result: lmfit.model.ModelResult, line_types: pd.DataFrame) -> str:
        fit_report = fit_result.fit_report(show_correl=False)
        param_legend = []
        for key in fit_result.best_values.keys():
            idx_param_name = key.split('_', 2)
            idx = int(idx_param_name[1])
            param_name = idx_param_name[2]
            legend = line_types.loc[idx].Legend
            param_legend.append((key, legend + ' ' + param_name))
        for old, new in param_legend:
            fit_report = fit_report.replace(old, new)
        if '[[Fit Statistics]]' in fit_report:
            idx = fit_report.find('[[Fit Statistics]]')
            fit_report = fit_report[idx:]
        return fit_report

    @asyncSlot()
    async def undo(self) -> None:
        if self.rs.time_start is None:
            self.rs.time_start = datetime.now()
        self.rs.time_start = datetime.now() - (datetime.now() - self.rs.time_start)
        self.rs.ui.statusBar.showMessage('Undo...' + self.text())
        info('Undo CommandAfterFitting')
        self.rs.ui.fit_params_table.model().set_dataframe(self.df_fit_params)
        self.rs.fitting.report_result = copy.deepcopy(self.report_result_old)
        if self.sigma3 is not None:
            self.rs.fitting.sigma3 = copy.deepcopy(self.sigma3)
        else:
            self.rs.fitting.sigma3.clear()
            self.rs.fitting.fill.setVisible(False)
        self.rs.ui.deconvoluted_dataset_table_view.model().set_dataframe(self.dataset_old)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        self.rs.fitting.redraw_curves_for_filename()
        self.rs.fitting.draw_sum_curve()
        self.rs.fitting.draw_residual_curve()
        self.rs.fitting.show_current_report_result()
        self.rs.fitting.update_sigma3_curves()
        self.rs.fitting.set_rows_visibility()
        self.rs.fitting.update_ignore_features_table()
        self.rs.ui.fit_params_table.model().sort_index()
        self.rs.ui.fit_params_table.model().model_reset_emit()
        # print('stop CommandAfterBatchFitting')
        info('stop CommandAfterBatchFitting')
        self.update_undo_redo_tooltips()
        time_end = self.rs.time_start
        if not self.rs.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.rs.time_start = None
        self.rs.ui.statusBar.showMessage('Batch fitting completed for ' + str(seconds) + ' sec.', 55000)
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandAfterGuess(QUndoCommand):
    """
    1. deconv_lines_table clear and add new from guess result
    2. fit_params_table clear and add params for ''
    3. update fit_report
    4. delete all lines from plot and create new
    5. update sum, residual, sigma3 data

    Parameters
    ----------
    mw : MainWindow
        Main window class
    result : list[ModelResult]
    line_type : str
        Gaussian, Lorentzian... etc.
    n_params : list[str]
        ['a', 'x0', 'dx'.... etc]
    description : str
        Description to set in tooltip
    """

    def __init__(self, mw, result: list[ModelResult], line_type: str, n_params: int, description: str = "Auto guess") \
            -> None:
        super(CommandAfterGuess, self).__init__(description)
        self._mw = mw
        self._results = result
        self._line_type = line_type
        self._n_params = n_params
        self._fit_report = ''
        self._df_lines_old = mw.ui.deconv_lines_table.model().dataframe().copy()
        self._df_params_old = mw.ui.fit_params_table.model().dataframe().copy()
        self.report_result_old = mw.fitting.report_result[''] if '' in mw.fitting.report_result else ''
        self.sigma3_old = self._mw.fitting.sigma3[''] if '' in self._mw.fitting.sigma3 else None
        self.setText(description)
        self.UndoAction = mw.action_undo
        self.RedoAction = mw.action_redo
        self.UndoStack = mw.undoStack

    @asyncSlot()
    async def redo(self) -> None:
        if self._mw.time_start is None:
            self._mw.time_start = datetime.now()
        self._mw.ui.statusBar.showMessage('Redo...' + self.text())
        info('redo CommandAfterGuess')

        self._mw.ui.deconv_lines_table.model().clear_dataframe()
        self._mw.ui.fit_params_table.model().clear_dataframe()
        self._mw.ui.report_text_edit.setText('')
        self._mw.fitting.remove_all_lines_from_plot()
        av_text, sigma3_top, sigma3_bottom = fitting_metrics(self._results)

        for fit_result in self._results:
            self.process_result(fit_result)
            self._fit_report += self.edited_fit_report(fit_result.fit_report(show_correl=False))
        if av_text:
            report_text = av_text + self._fit_report
        else:
            report_text = self._fit_report
        self._mw.fitting.report_result.clear()
        self._mw.fitting.report_result[''] = report_text
        self._mw.ui.report_text_edit.setText(report_text)
        self._mw.fitting.draw_sum_curve()
        self._mw.fitting.draw_residual_curve()
        x_axis, _ = self._mw.fitting.sum_array()
        if sigma3_top.shape[0] < x_axis.shape[0]:
            d = x_axis.shape[0] - sigma3_top.shape[0]
            zer = np.zeros(d)
            sigma3_top = np.concatenate((sigma3_top, zer))
        if sigma3_bottom.shape[0] < x_axis.shape[0]:
            d = x_axis.shape[0] - sigma3_bottom.shape[0]
            zer = np.zeros(d)
            sigma3_bottom = np.concatenate((sigma3_bottom, zer))
        self._mw.fitting.sigma3[''] = x_axis, sigma3_top, sigma3_bottom
        self._mw.fitting.update_sigma3_curves('')
        create_task(self._stop())

    @staticmethod
    def edited_fit_report(fit_report: str) -> str:
        if '[[Fit Statistics]]' in fit_report:
            idx = fit_report.find('[[Fit Statistics]]')
            fit_report = fit_report[idx:]
        fit_report += '\n' + '\n'
        fit_report = fit_report.replace('dot', '.')
        return fit_report

    @asyncSlot()
    async def undo(self) -> None:
        self._mw.time_start = datetime.now() - (datetime.now() - self._mw.time_start)
        self._mw.ui.statusBar.showMessage('Undo...' + self.text())
        info('Undo CommandAfterFitting, filename %s' % '')
        self._mw.ui.deconv_lines_table.model().set_dataframe(self._df_lines_old)
        self._mw.ui.fit_params_table.model().set_dataframe(self._df_params_old)
        self._mw.fitting.set_rows_visibility()
        self._mw.fitting.remove_all_lines_from_plot()
        self._mw.fitting.report_result[''] = self.report_result_old
        self._mw.fitting.show_current_report_result()
        if self.sigma3_old is not None:
            self._mw.fitting.sigma3[''] = self.sigma3_old[0], self.sigma3_old[1], self.sigma3_old[2]
        else:
            del self._mw.fitting.sigma3['']
            self._mw.fitting.fill.setVisible(False)
        self._mw.fitting.update_sigma3_curves('')
        await self._mw.fitting.draw_all_curves()
        create_task(self._stop())

    def process_result(self, fit_result: lmfit.model.ModelResult) -> None:
        params = fit_result.params
        idx = 0
        line_params = {}
        rnd_style = random_line_style()
        # add fit lines and fit parameters table rows
        for i, j in enumerate(fit_result.best_values.items()):
            legend_param = j[0].replace('dot', '.').split('_', 1)
            if i % self._n_params == 0:
                line_params = {}
                rnd_style = random_line_style()
                idx = self._mw.ui.deconv_lines_table.model().append_row(legend_param[0], self._line_type, rnd_style)
            line_params[legend_param[1]] = j[1]
            v = np.round(j[1], 5)
            min_v = np.round(params[j[0]].min, 5)
            max_v = np.round(params[j[0]].max, 5)
            self._mw.ui.fit_params_table.model().append_row(idx, legend_param[1], v, min_v, max_v)
            if i % self._n_params == self._n_params - 1:
                self._mw.fitting.add_deconv_curve_to_plot(line_params, idx, rnd_style, self._line_type)

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        info('stop CommandAfterGuess')
        self.update_undo_redo_tooltips()
        self._mw.fitting.set_rows_visibility()
        time_end = self._mw.time_start
        if not self._mw.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self._mw.time_start = None
        self._mw.ui.statusBar.showMessage('Guess completed for ' + str(seconds) + ' sec.', 55000)
        self._mw.set_modified()
        collect(2)
        await sleep(0)


class CommandFitIntervalAdded(QUndoCommand):
    """
    undo / redo add row to fit_intervals_table_view

    Parameters
    ----------
    rs
        Main window class
    description : str
        Description to set in tooltip
    """

    def __init__(self, rs, description: str) -> None:
        super(CommandFitIntervalAdded, self).__init__(description)
        info('init FitIntervalAdded {!s}:'.format(str(description)))
        self.setText(description)
        self.df = rs.ui.fit_intervals_table_view.model().dataframe()
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.rs = rs

    @asyncSlot()
    async def redo(self) -> None:
        debug('redo FitIntervalAdded')
        self.rs.ui.fit_intervals_table_view.model().append_row()
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        info('undo FitIntervalAdded')
        self.rs.ui.fit_intervals_table_view.model().set_dataframe(self.df)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.rs.set_buttons_ability()

    async def _stop(self) -> None:
        info('stop FitIntervalAdded')
        self.update_undo_redo_tooltips()
        # print('stop FitIntervalAdded')
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandFitIntervalDeleted(QUndoCommand):
    """
    undo / redo delete row to fit_intervals_table_view

    Parameters
    ----------
    rs
        Main window class
    interval_number: int
        row index
    description : str
        Description to set in tooltip
    """

    def __init__(self, rs, interval_number: int, description: str) -> None:
        super(CommandFitIntervalDeleted, self).__init__(description)
        info('init FitIntervalDeleted {!s}:'.format(str(description)))
        self.setText(description)
        self.interval_number = interval_number
        self.df = rs.ui.fit_intervals_table_view.model().dataframe()
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.rs = rs

    @asyncSlot()
    async def redo(self) -> None:
        debug('redo FitIntervalDeleted')
        self.rs.ui.fit_intervals_table_view.model().delete_row(self.interval_number)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        info('undo FitIntervalDeleted')
        self.rs.ui.fit_intervals_table_view.model().set_dataframe(self.df)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.rs.set_buttons_ability()

    async def _stop(self) -> None:
        info('stop FitIntervalDeleted')
        self.update_undo_redo_tooltips()
        # print('stop FitIntervalDeleted')
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandFitIntervalChanged(QUndoCommand):
    """
    undo / redo change value of fit_intervals_table_view row

    Parameters
    ----------
    rs
        Main window class
    index: QModelIndex
        index in table
    new_value: float
        value to write by index
    model: PandasModelFitParamsTable
        pandas model
    description : str
        Description to set in tooltip
    """

    def __init__(self, rs, index: QModelIndex, new_value: float, model,
                 description: str) -> None:
        super(CommandFitIntervalChanged, self).__init__(description)
        info('init FitIntervalChanged {!s}:'.format(str(description)))
        self.setText(description)
        self.index = index
        self.new_value = new_value
        self.model = model
        self.df = rs.ui.fit_intervals_table_view.model().dataframe().copy()
        self.UndoAction = rs.action_undo
        self.RedoAction = rs.action_redo
        self.UndoStack = rs.undoStack
        self.rs = rs

    @asyncSlot()
    async def redo(self) -> None:
        debug('redo FitIntervalChanged to {!s}'.format(self.new_value))
        self.model.setData(self.index, self.new_value, Qt.EditRole)
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        info('undo FitIntervalChanged {!s}'.format(self.df))
        self.rs.ui.fit_intervals_table_view.model().set_dataframe(self.df)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')
        self.rs.set_buttons_ability()

    async def _stop(self) -> None:
        info('stop FitIntervalChanged')
        self.update_undo_redo_tooltips()
        self.model.sort_by_border()
        # print('stop FitIntervalChanged')
        self.rs.set_modified()
        collect(2)
        await sleep(0)


class CommandAfterFittingStat(QUndoCommand):
    """
    undo / redo fit classificator model

    Parameters
    ----------
    main_window
        Main window class
    result: of fitting model
    cl_type: type of classificator
        value to write by index
    description : str
        Description to set in tooltip
    """

    def __init__(self, main_window, result: dict, cl_type: str, description: str) -> None:
        super(CommandAfterFittingStat, self).__init__(description)

        self.stat_result_new = None
        info('init FitIntervalChanged {!s}:'.format(str(description)))
        self.setText(description)
        self.result = result
        self.cl_type = cl_type
        if cl_type not in main_window.stat_analysis_logic.latest_stat_result:
            self.stat_result_old = None
        else:
            self.stat_result_old = copy.deepcopy(main_window.stat_analysis_logic.latest_stat_result[cl_type])
        self.UndoAction = main_window.action_undo
        self.RedoAction = main_window.action_redo
        self.UndoStack = main_window.undoStack
        self.mw = main_window
        self.stat_result_new = self.create_stat_result()

    def create_stat_result(self) -> dict:
        if self.cl_type == 'LDA':
            stat_result_new = self.create_stat_result_lda()
        elif self.cl_type == 'QDA':
            stat_result_new = self.create_stat_result_lda()
        elif self.cl_type == 'Logistic regression':
            stat_result_new = self.create_stat_result_lr()
        elif self.cl_type == 'NuSVC':
            stat_result_new = self.create_stat_result_svc()
        elif self.cl_type == 'Nearest Neighbors' or self.cl_type == 'GPC' or self.cl_type == 'Decision Tree' \
                or self.cl_type == 'Naive Bayes' \
                or self.cl_type == 'Random Forest' \
                or self.cl_type == 'AdaBoost' or self.cl_type == 'MLP' or self.cl_type == 'XGBoost':
            stat_result_new = self.create_stat_result_rf()
        elif self.cl_type == 'Torch':
            stat_result_new = self.create_stat_result_torch()
        elif self.cl_type == 'PCA':
            stat_result_new = self.create_stat_result_pca()
        elif self.cl_type == 'PLS-DA':
            stat_result_new = self.create_stat_result_plsda()
        if self.mw.ui.dataset_type_cb.currentText() == 'Smoothed':
            X_display = self.mw.ui.smoothed_dataset_table_view.model().dataframe()
        elif self.mw.ui.dataset_type_cb.currentText() == 'Baseline corrected':
            X_display = self.mw.ui.baselined_dataset_table_view.model().dataframe()
        else:
            X_display = self.mw.ui.deconvoluted_dataset_table_view.model().dataframe()
        stat_result_new['X_display'] = X_display
        return stat_result_new

    def create_stat_result_lda(self) -> dict:
        result = copy.deepcopy(self.result)
        if 'y_pred_2d' in result:
            y_pred_2d = result['y_pred_2d']
        else:
            y_pred_2d = None
        result['y_pred_2d'] = y_pred_2d
        X = result['X']
        y_score_dec_func = result['y_score_dec_func']
        label_binarizer = LabelBinarizer().fit(result['y_train'])
        result['y_onehot_test'] = label_binarizer.transform(result['y_test'])
        binary = False
        if result['features_in_2d'].shape[1] == 1:
            binary = True
        else:
            classifier = OneVsRestClassifier(make_pipeline(StandardScaler(), result['model']))
            classifier.fit(result['x_train'], result['y_train'])
            y_score_dec_func = classifier.decision_function(result['x_test'])
        metrics_result = model_metrics(result['y_test'], result['y_pred_test'], binary, result['target_names'])
        metrics_result['accuracy_score_train'] = result['accuracy_score_train']
        result['metrics_result'] = metrics_result
        result['y_train_plus_test'] = np.concatenate((result['y_train'], result['y_test']))
        result['y_score_dec_func'] = y_score_dec_func
        if not self.mw.ui.use_shapley_cb.isChecked():
            return result
        if self.cl_type == 'LDA':
            explainer = shap.Explainer(result['model'].best_estimator_, X)
            shap_values = explainer(X)
            shap_values_legacy = explainer.shap_values(X)
        else:
            explainer = shap.Explainer(result['model'].predict, X, max_evals=2 * len(result['feature_names']) + 1)
            shap_values = explainer(X)
            shap_values_legacy = explainer.__call__(X)
            explainer = None
        result['explainer'] = explainer
        result['shap_values'] = shap_values
        result['shap_values_legacy'] = shap_values_legacy

        return result

    def create_stat_result_lr(self) -> dict:
        y_test = self.result['y_test']
        result = copy.deepcopy(self.result)
        result['y_train_plus_test'] = np.concatenate((self.result['y_train'], y_test))
        binary = True if len(self.result['model'].classes_) == 2 else False
        metrics_result = model_metrics(y_test, self.result['y_pred_test'], binary, self.result['target_names'])
        metrics_result['accuracy_score_train'] = self.result['accuracy_score_train']
        result['metrics_result'] = metrics_result
        label_binarizer = LabelBinarizer().fit(self.result['y_train'])
        result['y_onehot_test'] = label_binarizer.transform(y_test)
        if not self.mw.ui.use_shapley_cb.isChecked():
            return result
        explainer = shap.Explainer(self.result['model'].best_estimator_, self.result['X'])
        result['shap_values'] = explainer(self.result['X'])
        result['explainer'] = explainer
        result['shap_values_legacy'] = explainer.shap_values(self.result['X'])
        return result

    def create_stat_result_svc(self) -> dict:
        y_test = self.result['y_test']
        result = copy.deepcopy(self.result)
        result['y_train_plus_test'] = np.concatenate((self.result['y_train'], y_test))
        binary = True if len(self.result['model'].classes_) == 2 else False
        metrics_result = model_metrics(y_test, self.result['y_pred_test'], binary, self.result['target_names'])
        metrics_result['accuracy_score_train'] = self.result['accuracy_score_train']
        result['metrics_result'] = metrics_result
        label_binarizer = LabelBinarizer().fit(self.result['y_train'])
        result['y_onehot_test'] = label_binarizer.transform(y_test)
        if not self.mw.ui.use_shapley_cb.isChecked():
            return result
        kernel_explainer = shap.KernelExplainer(self.result['model'].best_estimator_.predict_proba, self.result['X'])
        explainer = shap.Explainer(self.result['model'].best_estimator_, self.result['X'],
                                   max_evals=2 * len(self.result['feature_names']) + 1)
        result['shap_values'] = explainer(self.result['X'])
        result['explainer'] = kernel_explainer
        result['shap_values_legacy'] = kernel_explainer.shap_values(self.result['X'])
        return result

    def create_stat_result_rf(self) -> dict:
        result = copy.deepcopy(self.result)
        y_test = result['y_test']
        x_train = result['x_train']
        model = result['model']
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_
        result['y_train_plus_test'] = np.concatenate((result['y_train'], y_test))
        binary = True if len(model.classes_) == 2 else False
        metrics_result = model_metrics(y_test, result['y_pred_test'], binary, result['target_names'])
        metrics_result['accuracy_score_train'] = result['accuracy_score_train']
        result['metrics_result'] = metrics_result
        label_binarizer = LabelBinarizer().fit(result['y_train'])
        result['y_onehot_test'] = label_binarizer.transform(y_test)
        if not self.mw.ui.use_shapley_cb.isChecked():
            return result
        func = lambda x: model.predict_proba(x)[:, 1]
        med = x_train.median().values.reshape((1, x_train.shape[1]))
        explainer = shap.Explainer(func, med, max_evals=2 * len(result['feature_names']) + 1)
        kernel_explainer = shap.KernelExplainer(func, med)
        result['shap_values'] = explainer(result['X'])
        result['expected_value'] = kernel_explainer.expected_value
        result['shap_values_legacy'] = kernel_explainer.shap_values(result['X'])
        return result

    def create_stat_result_torch(self) -> dict:
        result = copy.deepcopy(self.result)
        y_test = result['y_test']
        x_train = result['x_train']
        model = result['model']
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_
        result['y_train_plus_test'] = np.concatenate((result['y_train'], y_test))
        binary = True if len(model.classes_) == 2 else False
        metrics_result = model_metrics(np.array(y_test) - 1, result['y_pred_test'], binary, result['target_names'])
        metrics_result['accuracy_score_train'] = result['accuracy_score_train']
        result['metrics_result'] = metrics_result
        label_binarizer = LabelBinarizer().fit(result['y_train'])
        result['y_onehot_test'] = label_binarizer.transform(y_test)
        if not self.mw.ui.use_shapley_cb.isChecked():
            return result
        func = lambda x: model.predict_proba(x.astype(np.float32))[:, 1]
        med = x_train.median().values.reshape((1, x_train.shape[1])).astype(np.float32)
        explainer = shap.Explainer(func, med, max_evals=2 * len(result['feature_names']) + 1)
        kernel_explainer = shap.KernelExplainer(func, med)
        result['shap_values'] = explainer(result['X'])
        result['expected_value'] = kernel_explainer.expected_value
        result['shap_values_legacy'] = kernel_explainer.shap_values(result['X'])
        return result

    def create_stat_result_pca(self) -> dict:
        result = copy.deepcopy(self.result)
        result['y_train_plus_test'] = np.concatenate((self.result['y_train'], self.result['y_test']))
        result['loadings'] = pd.DataFrame(result['model'].components_.T, columns=['PC1', 'PC2'],
                                          index=result['feature_names'])
        return result

    def create_stat_result_plsda(self) -> dict:
        result = copy.deepcopy(self.result)
        result['y_train_plus_test'] = np.concatenate((self.result['y_train'], self.result['y_test']))
        result['vips'] = calculate_vips(result['model'])
        return result

    @asyncSlot()
    async def redo(self) -> None:
        debug('redo CommandAfterFittingStat')
        if self.stat_result_new is not None:
            self.mw.stat_analysis_logic.latest_stat_result[self.cl_type] = self.stat_result_new
        create_task(self._stop())

    @asyncSlot()
    async def undo(self) -> None:
        info('undo CommandAfterFittingStat')
        self.mw.stat_analysis_logic.latest_stat_result[self.cl_type] = self.stat_result_old
        if self.stat_result_old is None:
            self.mw.clear_selected_step(self.cl_type)
        create_task(self._stop())

    def update_undo_redo_tooltips(self) -> None:
        if self.UndoStack.canUndo():
            self.UndoAction.setToolTip(self.text())
        else:
            self.UndoAction.setToolTip('')
        if self.UndoStack.canRedo():
            self.RedoAction.setToolTip(self.text())
        else:
            self.RedoAction.setToolTip('')

    async def _stop(self) -> None:
        if self.cl_type == 'PCA' and self.mw.stat_analysis_logic.latest_stat_result['PCA'] is not None:
            self.mw.loop.run_in_executor(None, self.mw.stat_analysis_logic.update_pca_plots)
        elif self.cl_type == 'PLS-DA' and self.mw.stat_analysis_logic.latest_stat_result['PLS-DA'] is not None:
            self.mw.loop.run_in_executor(None, self.mw.stat_analysis_logic.update_plsda_plots)
        elif self.cl_type != 'PCA' and self.cl_type != 'PLS-DA' \
                and self.mw.stat_analysis_logic.latest_stat_result[self.cl_type] is not None:
            self.mw.loop.run_in_executor(None, self.mw.stat_analysis_logic.update_plots, self.cl_type)

        self.mw.stat_analysis_logic.update_force_single_plots(self.cl_type)
        self.mw.stat_analysis_logic.update_force_full_plots(self.cl_type)
        self.mw.stat_analysis_logic.update_stat_report_text()
        info('stop CommandAfterFittingStat')
        self.update_undo_redo_tooltips()
        self.mw.set_modified()

        time_end = self.mw.time_start
        if not self.mw.time_start:
            time_end = datetime.now()
        seconds = round((datetime.now() - time_end).total_seconds())
        self.mw.time_start = None
        self.mw.ui.statusBar.showMessage('Model fitting completed for ' + str(seconds) + ' sec.', 550000)
        collect(2)
        await sleep(0)


def fitting_metrics(fit_results: list[ModelResult]) \
        -> tuple[str, np.ndarray, np.ndarray]:
    ranges = 0
    chisqr_av = []
    redchi_av = []
    aic_av = []
    bic_av = []
    rsquared_av = []
    sigma3_top = np.array([])
    sigma3_bottom = np.array([])
    av_text = ''
    for fit_result in fit_results:
        ranges += 1
        chisqr_av.append(fit_result.chisqr)
        redchi_av.append(fit_result.redchi)
        aic_av.append(fit_result.aic)
        bic_av.append(fit_result.bic)
        try:
            rsquared_av.append(fit_result.rsquared)
        except:
            debug("fit_result.rsquared error")
        dely = fit_result.eval_uncertainty(sigma=3)
        sigma3_top = np.concatenate((sigma3_top, fit_result.best_fit + dely))
        sigma3_bottom = np.concatenate((sigma3_bottom, fit_result.best_fit - dely))
    if ranges != 0:
        chisqr_av = np.round(np.mean(chisqr_av), 6)
        redchi_av = np.round(np.mean(redchi_av), 6)
        aic_av = np.round(np.mean(aic_av), 6)
        bic_av = np.round(np.mean(bic_av), 6)
        rsquared_av = np.round(np.mean(rsquared_av), 8)
        av_text = "[[Average Fit Statistics]]" + '\n' \
                  + f"    chi-square         = {chisqr_av}" + '\n' \
                  + f"    reduced chi-square = {redchi_av}" + '\n' \
                  + f"    Akaike info crit   = {aic_av}" + '\n' \
                  + f"    Bayesian info crit = {bic_av}" + '\n' \
                  + f"    R-squared          = {rsquared_av}" + '\n' + '\n'
    return av_text, sigma3_top, sigma3_bottom
