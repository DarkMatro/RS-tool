"""
This module defines a class for various preprocessing stages such as input data handling,
conversion, spectrum cutting, normalization, smoothing, and baseline correction.

Classes
-------
PreprocessingStage
    Abstract class for defining a stage for preprocessing data.
"""

from qtpy.QtWidgets import QFileDialog
from qtpy.QtCore import QObject
from src.data.collections import ObservableDict
from asyncqtpy import asyncSlot
from src.data.get_data import get_parent
from ....data.config import get_config
from pathlib import Path
from src.files.export import export_files


class PreprocessingStage(QObject):
    """
    Abstract class for defining a stage for preprocessing data.

    Attributes
    ----------
    parent : object
        The parent object.
    ui : object, optional
        User interface object.
    data : ObservableDict
        Dictionary to store the data.
    active : bool
        Status of the preprocessing stage.
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.ui = None
        self.data = ObservableDict()
        self.active = True

    def set_ui(self, ui: object) -> None:
        """
        Set user interface object.

        Parameters
        ----------
        ui : object
            The user interface widget.
        """
        self.ui = ui

    def reset(self) -> None:
        """
        Reset class data.

        Returns
        -------
        None
        """
        self.data.clear()

    def read(self) -> dict:
        """
        Read attributes data.

        Returns
        -------
        dict
            Dictionary containing all class attributes data.
        """
        dt = {"data": self.data.get_data()}
        return dt

    def load(self, db: dict) -> None:
        """
        Load attributes data from file.

        Parameters
        ----------
        db : dict
            Dictionary containing all class attributes data.

        Returns
        -------
        None
        """
        self.data.update(db['data'])

    @asyncSlot()
    async def save(self) -> None:
        """
        Save attributes data to file asynchronously.

        Returns
        -------
        None
        """
        mw = get_parent(self.parent, 'MainWindow')
        if not self.data:
            mw.ui.statusBar.showMessage('Export failed. No files to save.', 25000)
            return
        if mw.progress.time_start is not None:
            return
        fd = QFileDialog(mw)
        folder_path = fd.getExistingDirectory(
            mw, "Choose folder to export files", mw.latest_file_path
        )
        if not folder_path:
            return
        mw.latest_file_path = folder_path
        cfg = get_config('texty')['export_spectrum']
        mw.progress.open_progress(cfg)
        if not Path(folder_path).exists():
            Path(folder_path).mkdir(parents=True)
        n_files = len(self.data)
        kwargs = {'n_files': n_files}
        await mw.progress.run_in_executor('export_files', export_files, self.data.items(),
                                          *[folder_path], **kwargs)
        mw.progress.close_progress(cfg)
        mw.ui.statusBar.showMessage(f"Export completed. {n_files} new files created.", 50_000)
        mw.progress.time_start = None

    def activate(self, b: bool | None = None) -> None:
        """
        Activate or deactivate the preprocessing stage.

        Parameters
        ----------
        b : bool, optional
            If None, toggles the active status. If bool, sets the active status to the given value.

        Returns
        -------
        None
        """
        if b is None:
            self.active = not self.active
        else:
            self.active = b
        self.ui.content.setEnabled(self.active)
        self.ui.activate_btn.setChecked(self.active)
