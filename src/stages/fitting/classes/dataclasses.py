"""
dataclasses.py

This module defines data classes for managing various types of data used in spectral
analysis and decomposition. The data classes include information about plot curves, styles,
data, tables, and plotting parameters, providing a structured way to handle and pass data
through different stages of the analysis.
"""

import dataclasses

import numpy as np
from pyqtgraph import PlotCurveItem, FillBetweenItem, LinearRegionItem

from src.data.collections import NestedDefaultDict
from src.stages.fitting.classes.table_decomp_lines import TableDecompLines
from src.stages.fitting.classes.table_filenames import TableFilenames
from src.stages.fitting.classes.table_fit_borders import TableFitBorders
from src.stages.fitting.classes.table_params import TableParams


@dataclasses.dataclass
class Curves:
    """
    A data class to hold different types of plot curves.

    Attributes
    ----------
    data : PlotCurveItem or None
        The plot curve for the data.
    sum : PlotCurveItem or None
        The plot curve for the sum.
    residual : PlotCurveItem or None
        The plot curve for the residual.
    sigma3_fill : FillBetweenItem or None
        The plot item for the fill between sigma3 top and bottom.
    sigma3_top : PlotCurveItem
        The plot curve for the top of the sigma3 range.
    sigma3_bottom : PlotCurveItem
        The plot curve for the bottom of the sigma3 range.
    """
    data: PlotCurveItem | None
    sum: PlotCurveItem | None
    residual: PlotCurveItem | None
    sigma3_fill: FillBetweenItem | None
    sigma3_top: PlotCurveItem
    sigma3_bottom: PlotCurveItem


@dataclasses.dataclass
class Styles:
    """
    A data class to hold style information for different plot elements.

    Attributes
    ----------
    data : dict
        Style for the data plot.
    sum : dict
        Style for the sum plot.
    residual : dict
        Style for the residual plot.
    sigma3 : dict
        Style for the sigma3 plot.
    """
    data: dict
    sum: dict
    residual: dict
    sigma3: dict


@dataclasses.dataclass
class Data:
    """
    A data class to hold various pieces of data related to the analysis.

    Attributes
    ----------
    report_result : dict
        Results of the report.
    is_template : bool
        Flag indicating if the current data is a template.
    current_spectrum_name : str
        The name of the current spectrum.
    sigma3 : dict
        Sigma3 data.
    averaged_spectrum : np.ndarray
        Averaged spectrum data.
    params_stderr : NestedDefaultDict
        Standard error of parameters.
    all_ranges_clustered_x0_sd : list of np.ndarray or None
        Clustered x0 standard deviations for all ranges.
    """
    report_result: dict
    is_template: bool
    current_spectrum_name: str
    sigma3: dict
    averaged_spectrum: np.ndarray
    params_stderr: NestedDefaultDict
    all_ranges_clustered_x0_sd: list[np.ndarray] | None


@dataclasses.dataclass
class Tables:
    """
    A data class to hold different table data used in the analysis.

    Attributes
    ----------
    fit_borders : TableFitBorders
        Table of fit borders.
    filenames : TableFilenames
        Table of filenames.
    decomp_lines : TableDecompLines
        Table of decomposition lines.
    fit_params_table : TableParams
        Table of fit parameters.
    """
    fit_borders: TableFitBorders
    filenames: TableFilenames
    decomp_lines: TableDecompLines
    fit_params_table: TableParams


@dataclasses.dataclass
class Plotting:
    """
    A data class to hold plotting parameters and related data.

    Attributes
    ----------
    dragged_line_parameters : tuple or None
        Parameters of the currently dragged line.
    prev_dragged_line_parameters : tuple or None
        Parameters of the previously dragged line.
    intervals_data : dict or None
        Data for plotting intervals.
    linear_region : LinearRegionItem or None
        The linear region item for the plot.
    """
    dragged_line_parameters: tuple | None
    prev_dragged_line_parameters: tuple | None
    intervals_data: dict | None
    linear_region: LinearRegionItem | None
