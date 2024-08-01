# pylint: disable=too-many-lines, no-name-in-module, import-error, relative-beyond-top-level
# pylint: disable=unnecessary-lambda, invalid-name, redefined-builtin
"""
Module for handling baseline correction of spectral data.

This module provides the BaselineData class which manages baseline correction of spectral data
using various methods. It interfaces with the user through a UI form and handles data
manipulation, UI updates, and baseline correction computations.
"""

from copy import deepcopy, copy
from os import environ
from typing import ItemsView

import numpy as np
import pandas as pd
from asyncqtpy import asyncSlot
from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QMainWindow

from src import (baseline_methods, baseline_parameter_defaults, get_config, get_parent,
                 ObservableDict, Ui_BaselineForm, PreprocessingStage, UndoCommand)
from src.data.plotting import get_curve_plot_data_item


class BaselineData(PreprocessingStage):
    """
    Handles baseline correction of spectrum from the previous stage.

    Parameters
    ----------
    parent : Preprocessing
        Instance of the Preprocessing class.

    Attributes
    ----------
    ui : object
        User interface form.
    baseline_data : ObservableDict
        Dictionary to store baseline data.
    baseline_methods : dict
        Available baseline correction methods.
    current_method : str
        Current baseline correction method being used.
    fields : dict
        Fields in the UI form corresponding to different parameters.
    baseline_one_curve : np.ndarray
        Baseline data for a single curve.
    not_corrected_one_curve : np.ndarray
        Original data for a single curve before baseline correction.
    name : str
        Name of the class instance.
    """

    # pylint: disable=too-many-instance-attributes
    # Eight is reasonable in this case.
    def __init__(self, parent, *args, **kwargs):
        """
        Initialize BaselineData with the given parent and optional arguments.

        Parameters
        ----------
        parent : Preprocessing
            Instance of the Preprocessing class.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(parent, *args, **kwargs)
        self.ui = None
        self.baseline_data = ObservableDict()
        self.baseline_methods = baseline_methods()
        self.current_method = ''
        self.fields = None
        self.baseline_one_curve = None
        self.not_corrected_one_curve = None
        self.name = 'BaselineData'

    def set_ui(self, ui: Ui_BaselineForm) -> None:
        """
        Set the user interface object.

        Parameters
        ----------
        ui : Ui_BaselineForm
            User interface form widget.
        """
        context = get_parent(self.parent, "Context")
        defaults = get_config('defaults')
        self.ui = ui
        self.ui.reset_btn.clicked.connect(self.reset)
        self.ui.save_btn.clicked.connect(self.save)
        self.ui.activate_btn.clicked.connect(self.activate)
        self.ui.bl_cor_btn.clicked.connect(self.process_clicked)
        self._init_method_combo_box()
        self._init_cost_func_combo_box()
        self.ui.cost_func_comboBox.currentTextChanged.connect(context.set_modified)
        self.ui.rebuild_y_check_box.stateChanged.connect(context.set_modified)
        self.fields = {'alpha_factor': self.ui.alpha_factor_doubleSpinBox,
                       'eta': self.ui.eta_doubleSpinBox,
                       'half_window': self.ui.half_window_spinBox,
                       'fraction': self.ui.fraction_doubleSpinBox,
                       'tol': self.ui.grad_doubleSpinBox,
                       'lambda': self.ui.lambda_spinBox,
                       'min_length': self.ui.min_length_spinBox,
                       'max_iter': self.ui.n_iterations_spinBox,
                       'num_std': self.ui.num_std_doubleSpinBox,
                       'p': self.ui.p_doubleSpinBox,
                       'peak_ratio': self.ui.peak_ratio_doubleSpinBox,
                       'polynome_degree': self.ui.polynome_degree_spinBox,
                       'quantile': self.ui.quantile_doubleSpinBox,
                       'scale': self.ui.scale_doubleSpinBox,
                       'sections': self.ui.sections_spinBox,
                       'spline_degree': self.ui.spline_degree_spinBox}
        for field in self.fields.values():
            field.valueChanged.connect(context.set_modified)
        self.ui.alpha_factor_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'alpha_factor')
        self.ui.eta_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'eta')
        self.ui.half_window_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'half_window')
        self.ui.fraction_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'fraction')
        self.ui.grad_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'tol')
        self.ui.lambda_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'lambda')
        self.ui.min_length_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'min_length')
        self.ui.n_iterations_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'max_iter')
        self.ui.num_std_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'num_std')
        self.ui.p_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'p')
        self.ui.peak_ratio_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'peak_ratio')
        self.ui.polynome_degree_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'polynome_degree')
        self.ui.quantile_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'quantile')
        self.ui.scale_doubleSpinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'scale')
        self.ui.sections_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'sections')
        self.ui.spline_degree_spinBox.mouseDoubleClickEvent = lambda event: \
            self.reset_field(event, 'spline_degree')
        self.ui.method_comboBox.currentTextChanged.connect(self.method_changed)
        self.ui.method_comboBox.setCurrentText(defaults['bl_method_comboBox'])
        self.method_changed(defaults['bl_method_comboBox'])

    def reset(self) -> None:
        """
        Reset class data to default values.
        """
        self.data.clear()
        self.baseline_data.clear()
        defaults = get_config('defaults')
        for k, field in self.fields.items():
            field.setValue(defaults[k])
        self.ui.method_comboBox.setCurrentText(defaults['bl_method_comboBox'])
        self.ui.cost_func_comboBox.setCurrentText(defaults['cost_func_comboBox'])
        self.current_method = ''
        if self.parent.active_stage == self:
            self.parent.update_plot_item('BaselineData')
        self.activate(True)
        self.ui.rebuild_y_check_box.setChecked(False)
        self.baseline_one_curve = None
        self.not_corrected_one_curve = None

    def read(self, production_export: bool=False) -> dict:
        """
        Read the current state of the class attributes.

        Returns
        -------
        dict
            Dictionary containing all class attributes data.
        """
        dt = {"baseline_data": self.baseline_data.get_data(),
              'bl_method_comboBox': self.ui.method_comboBox.currentText(),
              'cost_func_comboBox': self.ui.cost_func_comboBox.currentText(),
              'current_method': self.current_method,
              'active': self.active}
        for k, field in self.fields.items():
            dt[k] = field.value()
        if not production_export:
            dt['data'] = self.data.get_data()
        return dt

    def load(self, db: dict) -> None:
        """
        Load class attributes data from a given dictionary.

        Parameters
        ----------
        db : dict
            Dictionary containing class attributes data.
        """
        if 'data' in db:
            self.data.update(db['data'])
        self.baseline_data.update(db['baseline_data'])
        self.ui.method_comboBox.setCurrentText(db['bl_method_comboBox'])
        self.ui.cost_func_comboBox.setCurrentText(db['cost_func_comboBox'])
        self.current_method = db['current_method']
        self.activate(db['active'])
        for k, field in self.fields.items():
            field.setValue(db[k])

    def reset_field(self, event: QMouseEvent, field_id: str) -> None:
        """
        Reset a specific field value to default on middle-button double-click.

        Parameters
        ----------
        event : QMouseEvent
            Mouse event triggering the reset.
        field_id : str
            Identifier of the field to reset.
        """
        if event.buttons() != Qt.MouseButton.MiddleButton:
            return
        method = self.ui.method_comboBox.currentText()
        defaults = baseline_parameter_defaults()
        if (method in defaults
                and field_id in defaults[method]):
            value = defaults[method][field_id]
        else:
            value = get_config('defaults')[field_id]
        assert field_id in self.fields, 'Something gone wrong. There is no such field_id.'
        field = self.fields[field_id]
        field.setValue(value)

    def plot_items(self) -> ItemsView:
        """
        Get data items for plotting.

        Returns
        -------
        ItemsView
            Items for plotting.
        """
        return self.data.items()

    def _init_method_combo_box(self) -> None:
        """
        Initialize the method combo box with available baseline methods.
        """
        self.ui.method_comboBox.addItems(self.baseline_methods.keys())

    def _init_cost_func_combo_box(self) -> None:
        """
        Initialize the cost function combo box with available cost functions.
        """
        self.ui.cost_func_comboBox.addItems(
            [
                "asymmetric_truncated_quadratic",
                "symmetric_truncated_quadratic",
                "asymmetric_huber",
                "symmetric_huber",
                "asymmetric_indec",
                "symmetric_indec",
            ]
        )

    def method_changed(self, current_text: str):
        """
        Handle changes in the selected baseline method.

        Parameters
        ----------
        current_text : str
            Currently selected method text.
        """
        get_parent(self.parent, "Context").set_modified()
        self.hide_all_field()
        match current_text:
            case "Poly":
                self.ui.polynome_degree_spinBox.setVisible(True)
            case "ModPoly":
                self._show_modpoly_fields()
            case "iModPoly":
                self._show_modpoly_fields()
                self.ui.num_std_doubleSpinBox.setVisible(True)
            case "ExModPoly":
                self._show_ex_mod_poly_fields()
            case "Penalized poly":
                self._show_modpoly_fields()
                self.ui.alpha_factor_doubleSpinBox.setVisible(True)
                self.ui.cost_func_comboBox.setVisible(True)
            case "LOESS":
                self._show_modpoly_fields()
                self.ui.fraction_doubleSpinBox.setVisible(True)
                self.ui.scale_doubleSpinBox.setVisible(True)
            case "Quantile regression":
                self._show_modpoly_fields()
                self.ui.quantile_doubleSpinBox.setVisible(True)
            case "Goldindec":
                self._show_modpoly_fields()
                self.ui.alpha_factor_doubleSpinBox.setVisible(True)
                self.ui.cost_func_comboBox.setVisible(True)
                self.ui.peak_ratio_doubleSpinBox.setVisible(True)
            case "AsLS" | "arPLS" | "airPLS":
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.p_doubleSpinBox.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
            case "iAsLS" | "psaLSA" | "DerPSALSA" | 'MPLS':
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.p_doubleSpinBox.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
            case "iarPLS" | 'asPLS':
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.grad_doubleSpinBox.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
            case "drPLS":
                self.ui.lambda_spinBox.setVisible(True)
                self.ui.p_doubleSpinBox.setVisible(True)
                self.ui.n_iterations_spinBox.setVisible(True)
                self.ui.eta_doubleSpinBox.setVisible(True)
            case 'Morphological' | 'Rolling Ball' | 'MWMV' | 'Top-hat':
                self.ui.half_window_spinBox.setVisible(True)
            case "iMor" | "MorMol" | "AMorMol":
                self._show_imor_fields()
            case "JBCD":
                self._show_jbcd_fields()
            case "MPSpline":
                self._show_mpspline_fields()
            case "Mixture Model":
                self._show_mixture_fields()
            case "IRSQR":
                self._show_irsqr_fields()
            case "Corner-Cutting":
                self.ui.n_iterations_spinBox.setVisible(True)
            case 'Noise Median' | 'SNIP' | 'SWiMA':
                self.ui.half_window_spinBox.setVisible(True)
            case 'IPSA' | 'RIA':
                self._show_ipsa_fields()
            case "Dietrich":
                self._show_dietrich_fields()
            case "Golotvin":
                self._show_golotvin_fields()
            case "Std Distribution":
                self._show_std_fields()
            case "FastChrom":
                self._show_fastchrom_fields()
            case "FABC":
                self._show_fabc_fields()
            case 'BEaDS':
                self._show_beads_fields()

    def hide_all_field(self) -> None:
        """
        Hide all UI fields.
        """
        self.ui.alpha_factor_doubleSpinBox.setVisible(False)
        self.ui.cost_func_comboBox.setVisible(False)
        self.ui.eta_doubleSpinBox.setVisible(False)
        self.ui.half_window_spinBox.setVisible(False)
        self.ui.fraction_doubleSpinBox.setVisible(False)
        self.ui.grad_doubleSpinBox.setVisible(False)
        self.ui.lambda_spinBox.setVisible(False)
        self.ui.min_length_spinBox.setVisible(False)
        self.ui.n_iterations_spinBox.setVisible(False)
        self.ui.num_std_doubleSpinBox.setVisible(False)
        self.ui.p_doubleSpinBox.setVisible(False)
        self.ui.polynome_degree_spinBox.setVisible(False)
        self.ui.quantile_doubleSpinBox.setVisible(False)
        self.ui.sections_spinBox.setVisible(False)
        self.ui.scale_doubleSpinBox.setVisible(False)
        self.ui.rebuild_y_check_box.setVisible(False)
        self.ui.spline_degree_spinBox.setVisible(False)
        self.ui.peak_ratio_doubleSpinBox.setVisible(False)

    def _show_modpoly_fields(self) -> None:
        """
        Show fields related to the ModPoly baseline method.
        """
        self.ui.n_iterations_spinBox.setVisible(True)
        self.ui.polynome_degree_spinBox.setVisible(True)
        self.ui.grad_doubleSpinBox.setVisible(True)

    def _show_ex_mod_poly_fields(self) -> None:
        """
        Show fields related to the ExModPoly baseline method.
        """
        self._show_modpoly_fields()
        self.ui.quantile_doubleSpinBox.setVisible(True)
        self.ui.scale_doubleSpinBox.setVisible(True)
        self.ui.rebuild_y_check_box.setVisible(True)
        self.ui.num_std_doubleSpinBox.setVisible(True)
        self.ui.half_window_spinBox.setVisible(True)

    def _show_imor_fields(self) -> None:
        """
        Show fields related to the iMor baseline method.
        """
        self.ui.n_iterations_spinBox.setVisible(True)
        self.ui.grad_doubleSpinBox.setVisible(True)
        self.ui.half_window_spinBox.setVisible(True)

    def _show_jbcd_fields(self) -> None:
        """
        Show fields related to the JBCD baseline method.
        """
        self.ui.n_iterations_spinBox.setVisible(True)
        self.ui.grad_doubleSpinBox.setVisible(True)
        self.ui.half_window_spinBox.setVisible(True)
        self.ui.alpha_factor_doubleSpinBox.setVisible(True)

    def _show_mpspline_fields(self) -> None:
        """
        Show fields related to the MPSpline baseline method.
        """
        self.ui.lambda_spinBox.setVisible(True)
        self.ui.p_doubleSpinBox.setVisible(True)
        self.ui.spline_degree_spinBox.setVisible(True)
        self.ui.half_window_spinBox.setVisible(True)

    def _show_mixture_fields(self) -> None:
        """
        Show fields related to the Mixture Model baseline method.
        """
        self.ui.lambda_spinBox.setVisible(True)
        self.ui.p_doubleSpinBox.setVisible(True)
        self.ui.n_iterations_spinBox.setVisible(True)
        self.ui.spline_degree_spinBox.setVisible(True)
        self.ui.grad_doubleSpinBox.setVisible(True)

    def _show_irsqr_fields(self) -> None:
        """
        Show fields related to the IRSQR baseline method.
        """
        self.ui.lambda_spinBox.setVisible(True)
        self.ui.quantile_doubleSpinBox.setVisible(True)
        self.ui.spline_degree_spinBox.setVisible(True)
        self.ui.n_iterations_spinBox.setVisible(True)
        self.ui.grad_doubleSpinBox.setVisible(True)

    def _show_ipsa_fields(self) -> None:
        """
        Show fields related to the IPSA baseline method.
        """
        self.ui.half_window_spinBox.setVisible(True)
        self.ui.n_iterations_spinBox.setVisible(True)
        self.ui.grad_doubleSpinBox.setVisible(True)

    def _show_dietrich_fields(self) -> None:
        """
        Show fields related to the Dietrich baseline method.
        """
        self._show_modpoly_fields()
        self.ui.num_std_doubleSpinBox.setVisible(True)
        self.ui.min_length_spinBox.setVisible(True)
        self.ui.half_window_spinBox.setVisible(True)

    def _show_golotvin_fields(self) -> None:
        """
        Show fields related to the Golotvin baseline method.
        """
        self.ui.num_std_doubleSpinBox.setVisible(True)
        self.ui.min_length_spinBox.setVisible(True)
        self.ui.half_window_spinBox.setVisible(True)
        self.ui.sections_spinBox.setVisible(True)

    def _show_std_fields(self) -> None:
        """
        Show fields related to the Std Distribution baseline method.
        """
        self.ui.num_std_doubleSpinBox.setVisible(True)
        self.ui.half_window_spinBox.setVisible(True)
        self.ui.half_window_spinBox.setVisible(True)

    def _show_fastchrom_fields(self) -> None:
        """
        Show fields related to the FastChrom baseline method.
        """
        self.ui.half_window_spinBox.setVisible(True)
        self.ui.min_length_spinBox.setVisible(True)
        self.ui.n_iterations_spinBox.setVisible(True)

    def _show_fabc_fields(self) -> None:
        """
        Show fields related to the FABC baseline method.
        """
        self.ui.lambda_spinBox.setVisible(True)
        self.ui.num_std_doubleSpinBox.setVisible(True)
        self.ui.min_length_spinBox.setVisible(True)

    def _show_beads_fields(self) -> None:
        """
        Show fields related to the BEaDS baseline method.
        """
        self.ui.lambda_spinBox.setVisible(True)
        self.ui.n_iterations_spinBox.setVisible(True)
        self.ui.grad_doubleSpinBox.setVisible(True)

    @asyncSlot()
    async def process_clicked(self) -> None:
        """
        Handle baseline correction button click event.
        """
        mw = get_parent(self.parent, "MainWindow")
        if mw.progress.time_start is not None:
            return
        prev_stage = mw.ui.drag_widget.get_previous_stage(self)
        if prev_stage is None or not prev_stage.data:
            mw.ui.statusBar.showMessage("No data for baseline correction")
            return
        await self._correction(mw, prev_stage.data)

    @asyncSlot()
    async def _correction(self, mw: QMainWindow, data: ObservableDict) -> None:
        """
        Perform baseline correction on the data.

        Parameters
        ----------
        mw : QMainWindow
            Main window instance.
        data : ObservableDict
            Dictionary containing the data to be corrected.
        """
        n_files = len(data)
        cfg = get_config("texty")["baseline"]
        method = self.ui.method_comboBox.currentText()
        mw.progress.open_progress(cfg, n_files)
        func, n_limit = self.baseline_methods[method]
        kwargs = {'n_files': n_files, 'n_limit': n_limit}
        kwargs.update(self.baseline_correction_params(method))
        result: list[tuple[str, np.ndarray, np.ndarray]] = await mw.progress.run_in_executor(
            "baseline", func, data.items(), **kwargs
        )
        cancel = mw.progress.close_progress(cfg)
        if cancel:
            return
        if not result:
            mw.ui.statusBar.showMessage(cfg["no_result_msg"])
        context = get_parent(self.parent, "Context")
        command = CommandBaseline(result, context, text="Baseline correction", **{'stage': self,
                                                                                  'method': method,
                                                                                  'params': kwargs})
        context.undo_stack.push(command)

    def baseline_correction_params(self, method: str) -> dict:
        """
        Get the baseline correction parameters for the given method.

        Parameters
        ----------
        method : str
            Baseline correction method.

        Returns
        -------
        dict
            Dictionary of parameters for the method.
        """
        params = {}
        match method:
            case 'Poly':
                params = {'poly_order': self.ui.polynome_degree_spinBox.value(),
                          'method': method}
            case 'ModPoly':
                params = {'poly_order': self.ui.polynome_degree_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'method': method}
            case 'iModPoly':
                params = {'poly_order': self.ui.polynome_degree_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'num_std': self.ui.num_std_doubleSpinBox.value(),
                          'method': method}
            case 'ExModPoly':
                params = {'poly_order': self.ui.polynome_degree_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'quantile': self.ui.quantile_doubleSpinBox.value(),
                          'w_scale_factor': self.ui.scale_doubleSpinBox.value(),
                          'recalc_y': self.ui.rebuild_y_check_box.isChecked(),
                          'num_std': self.ui.num_std_doubleSpinBox.value(),
                          'window_size': self.ui.half_window_spinBox.value()}
            case 'Penalized poly':
                params = {'poly_order': self.ui.polynome_degree_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'alpha_factor': self.ui.alpha_factor_doubleSpinBox.value(),
                          'cost_function': self.ui.cost_func_comboBox.currentText(),
                          'method': method, 'threshold': 0.001}
            case 'Goldindec':
                params = {'poly_order': self.ui.polynome_degree_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(), 'method': method,
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'alpha_factor': self.ui.alpha_factor_doubleSpinBox.value(),
                          'cost_function': self.ui.cost_func_comboBox.currentText(),
                          'peak_ratio': self.ui.peak_ratio_doubleSpinBox.value()}
            case 'Quantile regression':
                params = {'poly_order': self.ui.polynome_degree_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'quantile': self.ui.quantile_doubleSpinBox.value(),
                          'method': method}
            case 'AsLS' | 'arPLS' | 'airPLS':
                params = {'lam': self.ui.lambda_spinBox.value(),
                          'p': self.ui.p_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value()}
            case 'psaLSA' | 'iAsLS' | 'DerPSALSA' | 'MPLS':
                params = {'lam': self.ui.lambda_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'p': self.ui.p_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'method': method}
            case 'iarPLS' | 'asPLS':
                params = {'lam': self.ui.lambda_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'method': method}
            case 'drPLS':
                params = {'lam': self.ui.lambda_spinBox.value(),
                          'p': self.ui.p_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'eta': self.ui.eta_doubleSpinBox.value()}
            case 'Morphological' | 'Rolling Ball' | 'MWMV' | 'Top-hat':
                params = {'half_window': self.ui.half_window_spinBox.value(),
                          'method': method}
            case 'iMor' | 'MorMol' | 'AMorMol':
                params = {'tol': self.ui.grad_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'method': method, 'half_window': self.ui.half_window_spinBox.value()
                          }
            case 'MPSpline':
                params = {'lam': self.ui.lambda_spinBox.value(),
                          'half_window': self.ui.half_window_spinBox.value(),
                          'spline_degree': self.ui.spline_degree_spinBox.value(),
                          'p': self.ui.p_doubleSpinBox.value(), 'method': method}
            case 'JBCD':
                params = {'tol': self.ui.grad_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'method': method, 'half_window': self.ui.half_window_spinBox.value(),
                          'alpha': self.ui.alpha_factor_doubleSpinBox.value(),
                          }
            case 'Mixture Model':
                params = {'lam': self.ui.lambda_spinBox.value(),
                          'spline_degree': self.ui.spline_degree_spinBox.value(),
                          'p': self.ui.p_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'method': method}
            case 'IRSQR':
                params = {'lam': self.ui.lambda_spinBox.value(),
                          'spline_degree': self.ui.spline_degree_spinBox.value(),
                          'quantile': self.ui.quantile_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'method': method}
            case 'Corner-Cutting':
                params = {'max_iter': self.ui.n_iterations_spinBox.value(),
                          'method': method}
            case 'SNIP':
                params = {'max_half_window': self.ui.half_window_spinBox.value(),
                          'method': method}
            case 'Noise Median':
                params = {'half_window': self.ui.half_window_spinBox.value(),
                          'method': method}
            case 'IPSA' | 'RIA':
                params = {'half_window': self.ui.half_window_spinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'method': method}
            case 'SWiMA':
                params = {'min_half_window': self.ui.half_window_spinBox.value(),
                          'method': method}
            case 'Dietrich':
                params = {'num_std': self.ui.num_std_doubleSpinBox.value(),
                          'poly_order': self.ui.polynome_degree_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'max_iter': self.ui.n_iterations_spinBox.value(),
                          'smooth_half_window': self.ui.half_window_spinBox.value(),
                          'min_length': self.ui.min_length_spinBox.value(),
                          'method': method}
            case 'Golotvin':
                params = {'num_std': self.ui.num_std_doubleSpinBox.value(),
                          'half_window': self.ui.half_window_spinBox.value(),
                          'min_length': self.ui.min_length_spinBox.value(),
                          'sections': self.ui.sections_spinBox.value(),
                          'method': method}
            case 'Std Distribution':
                params = {'num_std': self.ui.num_std_doubleSpinBox.value(),
                          'half_window': self.ui.half_window_spinBox.value(),
                          'method': method}
            case 'FastChrom':
                params = {'max_iter': self.ui.n_iterations_spinBox.value(),
                          'half_window': self.ui.half_window_spinBox.value(),
                          'min_length': self.ui.min_length_spinBox.value(),
                          'method': method}
            case 'FABC':
                params = {'num_std': self.ui.num_std_doubleSpinBox.value(),
                          'min_length': self.ui.min_length_spinBox.value(),
                          'lam': self.ui.lambda_spinBox.value(),
                          'method': method}
            case 'BEaDS':
                params = {'max_iter': self.ui.n_iterations_spinBox.value(),
                          'tol': self.ui.grad_doubleSpinBox.value(),
                          'lam_0': self.ui.lambda_spinBox.value(),
                          'method': method}
        return params

    async def baseline_add_plot(self, current_spectrum_name: str) -> None:
        """
        Add baseline to plot

        Parameters
        ----------
        current_spectrum_name : str
            filename
        """
        # selected spectrum despiked
        mw = get_parent(self.parent, "MainWindow")
        arr = self.baseline_data[current_spectrum_name]
        arr_before = copy(arr)
        arr_before[:, 1] = arr[:, 1] + self.data[current_spectrum_name][:, 1]
        if self.baseline_one_curve:
            mw.ui.preproc_plot_widget.getPlotItem().removeItem(self.baseline_one_curve)
        if self.not_corrected_one_curve:
            mw.ui.preproc_plot_widget.getPlotItem().removeItem(self.not_corrected_one_curve)
        self.baseline_one_curve = get_curve_plot_data_item(arr, environ['primaryColor'])
        self.not_corrected_one_curve = (
            get_curve_plot_data_item(arr_before, environ['secondaryLightColor']))
        mw.ui.preproc_plot_widget.getPlotItem().addItem(self.baseline_one_curve,
                                                        kargs=['ignoreBounds', 'skipAverage'])
        mw.ui.preproc_plot_widget.getPlotItem().addItem(self.not_corrected_one_curve,
                                                        kargs=['ignoreBounds', 'skipAverage'])

    async def baseline_remove_plot(self) -> None:
        """
        Remove old history _BeforeDespike plot item and arrows
        """
        mw = get_parent(self.parent, "MainWindow")
        plot_item = mw.ui.preproc_plot_widget.getPlotItem()
        if self.baseline_one_curve:
            plot_item.removeItem(self.baseline_one_curve)
        if self.not_corrected_one_curve:
            plot_item.removeItem(self.not_corrected_one_curve)


class CommandBaseline(UndoCommand):
    """
    Change data for baseline correction stage.

    Parameters
    -------
    data: list[tuple[str, ndarray]]
        filename: str
            as input
        array: np.ndarray
            processed 2D array with normalized wavelengths and intensities
    parent: Context
        Backend context class
    text: str
        description
    """

    def __init__(self, data: list[tuple[str, np.ndarray, np.ndarray]],
                 parent, text: str, *args, **kwargs) -> None:
        self.stage = kwargs.pop('stage')
        method = kwargs.pop('method')
        params = kwargs.pop('params')
        super().__init__(data, parent, text, *args, **kwargs)
        self.bl_corrected_data = {k: y_new for k, _, y_new in data}
        self.bl_data = {k: bl for k, bl, _ in data}
        self.bl_corrected_data_old = deepcopy(self.stage.data.get_data())
        self.bl_data_old = deepcopy(self.stage.baseline_data.get_data())
        self.method_name = {'new': self.generate_title_text(method, params),
                            'old': copy(self.stage.current_method)}
        self.bl_df = {'new': deepcopy(self.create_baseline_corrected_dataset_new()),
                      'old': deepcopy(self.mw.ui.baselined_dataset_table_view.model().dataframe)}

    def redo_special(self):
        """
        Update data
        """
        self.stage.data.clear()
        self.stage.baseline_data.clear()
        self.stage.data.update(self.bl_corrected_data)
        self.stage.baseline_data.update(self.bl_data)
        self.stage.current_method = self.method_name['new']
        self.mw.ui.baselined_dataset_table_view.model().set_dataframe(self.bl_df['new'])

    def undo_special(self):
        """
        Undo data
        """
        self.stage.data.clear()
        self.stage.baseline_data.clear()
        self.stage.data.update(self.old_data)
        self.stage.baseline_data.update(self.bl_data_old)
        self.stage.current_method = self.method_name['old']
        self.mw.ui.baselined_dataset_table_view.model().set_dataframe(self.bl_df['old'])

    def stop_special(self) -> None:
        """
        Update ui elements.
        """
        self.parent.preprocessing.update_plot_item("BaselineData")

    def generate_title_text(self, method: str, params: dict) -> str:
        """
        Create title text for plot

        Parameters
        -------
        method: str

        params: dict
            all used parameters

        Returns
        -------
        text: str
        """
        text = method + '. '
        for param_name, value in params.items():
            text += param_name + ': ' + str(value) + '. '
        return text

    def create_baseline_corrected_dataset_new(self) -> pd.DataFrame:
        """
        Create dataframe for baseline corrected table.

        Returns
        -------
        df: pd.DataFrame:
        """
        filename_group = self.mw.ui.input_table.model().column_data(2)
        x_axis = next(iter(self.bl_corrected_data.values()))[:, 0]
        columns_params = [f'k{np.round(i, 2)}' for i in x_axis]
        df = pd.DataFrame(columns=columns_params)
        class_ids = []
        for filename, n_array in self.bl_corrected_data.items():
            class_ids.append(filename_group.loc[filename])
            df2 = pd.DataFrame(n_array[:, 1].reshape(1, -1), columns=columns_params)
            df = pd.concat([df, df2], ignore_index=True)
        df2 = pd.DataFrame({'Class': class_ids, 'Filename': list(filename_group.index)})
        df = pd.concat([df2, df], axis=1)
        return df
