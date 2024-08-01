# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Input data with import, interpolate, and despike functionalities.

Classes
-------
InputData : PreprocessingStage
    Handles importing files, interpolation, and despiking of data.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from asyncqtpy import asyncSlot
from pyqtgraph import ArrowItem
from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QFileDialog, QMainWindow

from src import get_config
from src.backend.undo_stack import UndoCommand
from src.data.collections import ObservableDict
from src.data.get_data import get_parent
from src.data.plotting import get_curve_plot_data_item
from src.data.work_with_arrays import nearest_idx
from src.ui.ui_import_widget import Ui_ImportForm
from .stages import PreprocessingStage
from ..functions.despiking import despike
from ..functions.importing import import_spectrum
from ..functions.interpolating import interpolate


class InputData(PreprocessingStage):
    """
    Class for importing files, interpolating, and despiking data.

    Parameters
    ----------
    parent : Preprocessing
        Instance of the Preprocessing class.

    Attributes
    ----------
    ui : Ui_ImportForm
        User interface form.
    data : ObservableDict
        Dictionary where keys are filenames and values are 2D array spectra.
    before_despike_data : ObservableDict
        Dictionary to compare data before and after despiking.
    ranges : dict
        Spectral range for interpolation.
    despiked_one_curve : PlotDataItem
        PyQtGraph plot item for showing spectrum before despiking.
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.ui = None
        self.data.on_change(self.data_changed)
        self.before_despike_data = ObservableDict()
        self.ranges = {}
        self.despiked_one_curve = None
        self.name = 'InputData'

    def data_changed(self, _):
        """
        Update selectable ranges for interpolation.
        Enable interpolation button if there are more than one range.
        Enable despike if there is any spectrum.
        """
        ranges = self._check_ranges()
        self.ranges = {str(k): k for k in ranges.keys()}
        self.ui.interp_combo_box.clear()
        self.ui.interp_combo_box.addItems(self.ranges.keys())
        if len(ranges) > 0:
            self.ui.interp_combo_box.setCurrentText(str(max(ranges)))
        enable = len(ranges) > 1
        self.ui.interp_combo_box.setEnabled(enable)
        self.ui.interp_btn.setEnabled(enable)
        self.ui.despike_gb.setEnabled(len(self.data) > 0)

    def reset(self) -> None:
        """
        Reset class data to default values.
        """
        self.data.clear()
        self.before_despike_data.clear()
        defaults = get_config('defaults')
        self.ui.order_spin_box.setValue(defaults['order_spin_box'])
        self.ui.spike_fwhm_spin_box.setValue(defaults['spike_fwhm_spin_box'])

    def read(self, production_export: bool=False) -> dict:
        """
        Read attributes data.

        Returns
        -------
        dict
            Dictionary containing all class attributes data.
        """
        dt = {'before_despike_data': self.before_despike_data.get_data(),
              'order_spin_box': self.ui.order_spin_box.value(),
              'spike_fwhm_spin_box': self.ui.spike_fwhm_spin_box.value()}
        if not production_export:
            dt['data'] = self.data.get_data()
        return dt

    def load(self, db: dict) -> None:
        """
        Load attributes data from file.

        Parameters
        ----------
        db : dict
            Dictionary containing all class attributes data.
        """
        if 'data' in db:
            self.data.update(db['data'])
        self.before_despike_data.update(db['before_despike_data'])
        self.ui.order_spin_box.setValue(db['order_spin_box'])
        self.ui.spike_fwhm_spin_box.setValue(db['spike_fwhm_spin_box'])

    def set_ui(self, ui: Ui_ImportForm) -> None:
        """
        Set the user interface as Ui_ImportForm.
        Setup buttons and input forms events.

        Parameters
        ----------
        ui : Ui_ImportForm
            User interface form widget.
        """
        context = get_parent(self.parent, "Context")
        self.ui = ui
        self.ui.import_btn.clicked.connect(self._importfile_clicked)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.interp_combo_box.setEnabled(False)
        self.ui.interp_btn.setEnabled(False)
        self.ui.interp_btn.clicked.connect(self.interpolate_clicked)
        self.ui.despike_gb.setEnabled(False)
        self.ui.despike_btn.clicked.connect(self.despike_clicked)
        self.ui.order_spin_box.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'order_spin_box')
        self.ui.spike_fwhm_spin_box.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'spike_fwhm_spin_box')
        self.ui.excel_btn.clicked.connect(self._save_as_excel)
        self.ui.order_spin_box.valueChanged.connect(context.set_modified)
        self.ui.spike_fwhm_spin_box.valueChanged.connect(context.set_modified)

    def reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Change field value to default on double-click with MiddleButton.

        Parameters
        ----------
        event : QMouseEvent
            Mouse event.
        field_id : str
            Name of the field to reset.
        """
        if event.buttons() != Qt.MouseButton.MiddleButton:
            return
        value = get_config('defaults')[field_id]
        match field_id:
            case 'order_spin_box':
                self.ui.order_spin_box.setValue(value)
            case 'spike_fwhm_spin_box':
                self.ui.spike_fwhm_spin_box.setValue(value)
            case _:
                return

    def plot_items(self) -> dict:
        """
        Get data for plotting.

        Returns
        -------
        dict
            Dictionary of items for plotting.
        """
        return self.data.items()

    @asyncSlot()
    async def _importfile_clicked(self) -> None:
        """
        Handle import file button clicked event to import files with Raman data.
        """
        main_window = get_parent(self.parent, "MainWindow")
        if main_window.progress.time_start is not None:
            return
        fd = QFileDialog(main_window)
        file_path = fd.getOpenFileNames(
            parent=main_window,
            caption="Select files with Raman data",
            directory=main_window.attrs.latest_file_path,
            filter="Text files (*.txt *.asc)",
        )
        if not file_path[0]:
            main_window.ui.statusBar.showMessage("No files were selected")
            return
        # exclude filename existing in self.data
        if self.data is None:
            filenames = list(file_path[0])
        else:
            filenames = [x for x in file_path[0] if Path(x).name not in self.data]
        main_window.attrs.latest_file_path = str(Path(filenames[0]).parent)
        await self.import_files(filenames)

    @asyncSlot()
    async def import_files(self, path_list: list[str]) -> None:
        """
        Import files with Raman data.

        Parameters
        ----------
        path_list : list of str
            List of selected filenames.
        """
        mw = get_parent(self.parent, "MainWindow")
        path_list = [x for x in path_list if Path(x).suffix.lower() in ['.txt', '.asc']]
        if len(path_list) == 0:
            return
        n_files = len(path_list)
        cfg = get_config("texty")["import"]

        mw.progress.open_progress(cfg, n_files)
        args = [self.parent.stages.convert_data.ui.laser_wl_spinbox.value()]
        kwargs = {'n_files': n_files}
        result = await mw.progress.run_in_executor(
            "import", import_spectrum, path_list, *args, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        context = get_parent(self.parent, "Context")
        command = CommandImportFiles(result, context, text="Import")
        context.undo_stack.push(command)

    @asyncSlot()
    async def _save_as_excel(self) -> None:
        """
        Action generates pandas table and save it into .csv format
        Table consists is like

        Group_id         | Filename | x_nm_0        | x_nm_1 .... x_nm_n
        int                 str      y_intensity_0  ....  y_intensity_n
        """
        mw = get_parent(self.parent, 'MainWindow')
        if not self.data:
            mw.ui.statusBar.showMessage('Export failed. No files to save.', 25000)
            return
        if mw.progress.time_start is not None:
            return
        fd = QFileDialog(mw)
        file_path = fd.getSaveFileName(
            mw, "Save input nm. data into csv table", mw.attrs.latest_file_path, "CSV (*.csv)")
        if not file_path[0]:
            return
        mw.attrs.latest_file_path = file_path[0]

        x_axis = next(iter(self.data.values()))[:, 0]
        nm_params = [str(np.round(i, 2)) for i in x_axis]
        df = pd.DataFrame(columns=nm_params)
        class_ids = []
        filename_group = mw.ui.input_table.model().column_data(2)
        for filename, n_array in self.data.items():
            class_ids.append(filename_group.loc[filename])
            df2 = pd.DataFrame(n_array[:, 1].reshape(1, -1), columns=nm_params)
            df = pd.concat([df, df2], ignore_index=True)
        indexes = list(filename_group.index)
        df2 = pd.DataFrame({'Class': class_ids, 'Filename': indexes})
        df = pd.concat([df2, df], axis=1)
        df.to_csv(file_path[0])

        mw.progress.time_start = None

    def _check_ranges(self) -> Counter:
        """
        Check ranges for interpolation.

        Returns
        -------
        Counter of ranges.
        """
        cnt = Counter()
        for _, arr in self.data.items():
            arr = arr[:, 0]
            new_range = (arr[0], arr[-1])
            cnt[new_range] += 1
        return cnt

    @asyncSlot()
    async def interpolate_clicked(self) -> None:
        """
        Handle interpolate button clicked event.
        """
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        if mw.progress.time_start is not None or len(self.data) < 2:
            return
        range_nm = self.ranges[self.ui.interp_combo_box.currentText()]
        if context.predict.is_production_project:
            reference_x_axis = context.predict.interp_ref_x_axis
        else:
            reference_x_axis = self._get_ref_x_axis(range_nm)
            context.predict.interp_ref_x_axis = reference_x_axis
        if reference_x_axis is None:
            return
        filenames = {k for k, spectrum in self.data.items()
                     if not np.array_equal(spectrum[:, 0], reference_x_axis)}
        n_files = len(filenames)
        cfg = get_config("texty")["interpolate"]
        mw.progress.open_progress(cfg, n_files)
        args = [reference_x_axis]
        data_to_interpolate = {k: v for k, v in self.data.items() if k in filenames}
        kwargs = {'n_files': n_files}
        result: list[tuple[str, np.ndarray]] = await mw.progress.run_in_executor(
            "interpolate", interpolate, data_to_interpolate.items(), *args, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        context = get_parent(self.parent, "Context")
        command = CommandInterpolate(result, context, text="Interpolate")
        context.undo_stack.push(command)

    def _get_ref_x_axis(self, target_range_nm: tuple[int, int]) -> np.ndarray | None:
        """
        Get reference X-axis for interpolation.

        Parameters
        ----------
        target_range_nm : tuple of int
            Target range in nm.

        Returns
        -------
        np.ndarray or None
            Reference X-axis array or None.
        """
        for spectrum in self.data.values():
            x_axis = spectrum[:, 0]
            new_range = (x_axis[0], x_axis[-1])
            if new_range == target_range_nm:
                return x_axis
        return None

    @asyncSlot()
    async def despike_clicked(self) -> None:
        """
        Handle despike button clicked event.
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None or len(self.data) == 0:
            return
        laser_wavelength = mw.context.preprocessing.stages.convert_data.ui.laser_wl_spinbox.value()
        maxima_count = self.ui.order_spin_box.value()
        fwhm = self.ui.spike_fwhm_spin_box.value()
        items_to_despike = self._get_items_to_despike(mw)
        n_files = len(items_to_despike)
        cfg = get_config("texty")["despike"]
        mw.progress.open_progress(cfg, n_files)
        args = [laser_wavelength, maxima_count, fwhm]
        kwargs = {'n_files': n_files}
        result: list[tuple[str, np.ndarray, list[float]]] = await mw.progress.run_in_executor(
            "despike", despike, items_to_despike, *args, **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return

        # Drop None results.
        result = [i for i in result if i]
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        context = get_parent(self.parent, "Context")
        command = CommandDespike(result, context, text="Despike")
        context.undo_stack.push(command)

    def _get_items_to_despike(self, mw: QMainWindow) -> list[tuple[str, np.ndarray, float]]:
        """
        Returns list of tuples as (filename, spectrum).
        If one spectrum selected - returns only this one spectrum
        If group of spectra selected - returns spectra of this group
        else returns all data items.

        Parameters
        -------
        mw: QMainWindow
            MainWindow class instance

        Returns
        -------
        items: list[tuple[str, np.ndarray, float]]
            passing to despike procedure (filename, arr, fwhm nm)
        """
        context = get_parent(self.parent, "Context")
        current_row = mw.ui.input_table.selectionModel().currentIndex().row()
        if mw.ui.by_one_control_button.isChecked() and current_row != -1:
            f_name = mw.ui.input_table.model().get_filename_by_row(current_row)
            items = [(f_name, self.data[f_name])]
        elif mw.ui.by_group_control_button.isChecked() \
                and context.group_table.selectionModel().currentIndex().row() != -1:
            group_id = context.group_table.selectionModel().currentIndex().row() + 1
            filenames = mw.ui.input_table.model().filenames_of_group_id(group_id)
            items = [i for i in self.data.items() if i[0] in filenames]
        else:
            items = self.data.items()
        fwhm_nm_df = mw.ui.input_table.model().get_column('FWHM, nm')
        result = []
        for filename, arr in items:
            result.append((filename, arr, fwhm_nm_df[filename]))
        return result

    async def despike_history_add_plot(self, current_spectrum_name) -> None:
        """
        Add arrows and before despike plot item to imported plot for comparison.
        """
        # selected spectrum despiked
        mw = get_parent(self.parent, "MainWindow")
        context = get_parent(self.parent, "Context")
        current_index = mw.ui.input_table.selectionModel().currentIndex()
        group_number = mw.ui.input_table.model().cell_data(current_index.row(), 2)
        arr = self.before_despike_data[current_spectrum_name]
        if self.despiked_one_curve:
            mw.ui.preproc_plot_widget.getPlotItem().removeItem(self.despiked_one_curve)
        color = context.group_table.get_color_by_group_number(group_number)
        self.despiked_one_curve = get_curve_plot_data_item(arr, color)
        mw.ui.preproc_plot_widget.getPlotItem().addItem(self.despiked_one_curve,
                                                        kargs=['ignoreBounds', 'skipAverage'])

        all_peaks = mw.ui.input_table.model().cell_data(current_index.row(), 3)
        all_peaks = all_peaks.split()
        text_peaks = []
        for i in all_peaks:
            i = i.replace(',', '')
            i = i.replace(' ', '')
            text_peaks.append(i)
        list_peaks = [float(s) for s in text_peaks]
        for i in list_peaks:
            idx = nearest_idx(arr[:, 0], i)
            y_peak = arr[:, 1][idx]
            arrow = ArrowItem(pos=(i, y_peak), angle=-45)
            mw.ui.preproc_plot_widget.getPlotItem().addItem(arrow)

    async def despike_history_remove_plot(self) -> None:
        """
        Remove old history before despike plot item and arrows.
        """
        print('despike_history_remove_plot')
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.preproc_plot_widget.getPlotItem()
        if self.despiked_one_curve:
            plot_item.removeItem(self.despiked_one_curve)

        arrows = []
        for x in plot_item.items:
            if isinstance(x, ArrowItem):
                arrows.append(x)
        for i in reversed(arrows):
            plot_item.removeItem(i)


class CommandImportFiles(UndoCommand):
    """
    Command for storing new imported spectra into InputData.

    Parameters
    ----------
    data : list of tuple
        List containing imported data information.
    parent : Context
        Backend context class.
    text : str
        Description of the command.
    """

    def __init__(self, data: list[tuple[str, np.ndarray, str, str, str, float]], parent,
            text: str, *args, **kwargs) -> None:
        super().__init__(data, parent, text, *args, **kwargs)
        self.cols = list(zip(*self.data))

    def redo_special(self):
        """
        Update input table and input data.
        """
        col_data = {
            "Min, nm": self.cols[3],
            "Max, nm": self.cols[4],
            "Group": self.cols[2],
            "FWHM, nm": self.cols[5],
        }
        self.mw.ui.input_table.model().concat_df_input_table(self.cols[0], col_data)
        upd = {}
        for filename, arr in zip(self.cols[0], self.cols[1]):
            upd[filename] = arr
        self.parent.preprocessing.stages.input_data.data.update(upd)

    def undo_special(self):
        """
        Undo update to input table and input data.
        """
        self.mw.ui.input_table.model().delete_rows(names=self.cols[0])
        old_data = self.parent.preprocessing.stages.input_data.data
        old_keys = set(self.cols[0])
        new_data = {k: v for k, v in old_data.items() if k not in old_keys}
        self.parent.preprocessing.stages.input_data.data.clear()
        self.parent.preprocessing.stages.input_data.data.update(new_data)

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.mw.ui.input_table.move(0, 1)
        self.parent.preprocessing.update_plot_item("InputData")
        self.parent.input_table.input_table_rows_changed()
        self.mw.ui.input_table.move(0, -1)
        self.mw.decide_vertical_scroll_bar_visible()


class CommandInterpolate(UndoCommand):
    """
    Command for storing interpolated spectra into InputData.

    Parameters
    ----------
    data : list of tuple
        List containing interpolated data information.
    parent : Context
        Backend context class.
    text : str
        Description of the command.
    """

    def __init__(
            self,
            data: list[tuple[str, np.ndarray]],
            parent,
            text: str,
            *args,
            **kwargs
    ) -> None:
        super().__init__(data, parent, text, *args, **kwargs)
        self.new_data = dict(data)
        self.old_data = {k: v for k, v in self.parent.preprocessing.stages.input_data.data.items()
                         if k in self.new_data.keys()}

    def redo_special(self):
        """
        Update input table and input data with interpolated values.
        """
        self.parent.preprocessing.stages.input_data.data.update(self.new_data)
        self.change_cells_input_table(self.new_data)

    def undo_special(self):
        """
        Revert input table and input data to previous state before interpolation.
        """
        self.parent.preprocessing.stages.input_data.data.update(self.old_data)
        self.change_cells_input_table(self.old_data)

    def change_cells_input_table(self, data: dict):
        """
        Update data in input table.

        Parameters
        ----------
        data : dict
            new or old data
        """
        model = self.mw.ui.input_table.model()
        for k, spectrum in data.items():
            x_axis = spectrum[:, 0]
            model.change_cell_data(k, 'Min, nm', x_axis[0])
            model.change_cell_data(k, 'Max, nm', x_axis[-1])

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.parent.preprocessing.update_plot_item("InputData")


class CommandDespike(UndoCommand):
    """
    Change data for despiked spectra.
    Update before_despike data.
    Update column 'Despiked, nm' in input plot.
    Refresh plots.

    Parameters
    -------
    data: list[tuple[str, np.ndarray, list[float]]
        filename: str
            as input
        return_array: np.ndarray
            fixed spectrum
        subtracted_peaks: list
            Wavelength array where peaks were fixed
    parent: Context
        Backend context class
    text: str
        description
    """

    def __init__(self, data: list[tuple[str, np.ndarray, list[float]]], parent, text: str, *args,
                 **kwargs) -> None:
        super().__init__(data, parent, text, *args, **kwargs)
        self.input_data = self.parent.preprocessing.stages.input_data
        self.new_data = {k: arr for k, arr, _ in data}
        self.new_peaks_text = {k: peaks for k, _, peaks in data}
        self.old_data = {k: v for k, v in self.input_data.data.items() if k in self.new_data.keys()}
        self.previous_text = {k: self.mw.ui.input_table.model().cell_data(k, 'Despiked, nm')
                              for k in self.new_data.keys()}
        self.before_despike_old_keys = self.input_data.before_despike_data.keys()

    def redo_special(self):
        """
        Update input table and input data with despiked values.
        """
        self.input_data.data.update(self.new_data)
        for k, arr in self.old_data.items():
            if k not in self.input_data.before_despike_data:
                self.input_data.before_despike_data[k] = arr
        for k in self.new_data.keys():
            new_text = str(self.new_peaks_text[k])
            new_text = new_text.replace('[', '').replace(']', '')
            previous_text = self.previous_text[k]
            if previous_text != '':
                new_text = previous_text + ', ' + new_text
            self.mw.ui.input_table.model().change_cell_data(k, 'Despiked, nm', new_text)

    def undo_special(self):
        """
         Revert input table and input data to previous state before despiking.
         """
        self.input_data.data.update(self.old_data)
        for k in self.new_data.keys():
            if k not in self.before_despike_old_keys:
                del self.input_data.before_despike_data[k]
        for k, v in self.previous_text.items():
            self.mw.ui.input_table.model().change_cell_data(k, 'Despiked, nm', v)

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.parent.preprocessing.update_plot_item("InputData")
