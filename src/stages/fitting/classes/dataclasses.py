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
    data: PlotCurveItem | None
    sum: PlotCurveItem | None
    residual: PlotCurveItem | None
    sigma3_fill: FillBetweenItem | None
    sigma3_top: PlotCurveItem
    sigma3_bottom: PlotCurveItem


@dataclasses.dataclass
class Styles:
    data: dict
    sum: dict
    residual: dict
    sigma3: dict


@dataclasses.dataclass
class Data:
    report_result: dict
    is_template: bool
    current_spectrum_name: str
    sigma3: dict
    averaged_spectrum: np.ndarray
    params_stderr: NestedDefaultDict
    all_ranges_clustered_x0_sd: list[np.ndarray] | None


@dataclasses.dataclass
class Tables:
    fit_borders: TableFitBorders
    filenames: TableFilenames
    decomp_lines: TableDecompLines
    fit_params_table: TableParams


@dataclasses.dataclass
class Plotting:
    dragged_line_parameters: tuple | None
    prev_dragged_line_parameters: tuple | None
    intervals_data: dict | None
    linear_region: LinearRegionItem | None
