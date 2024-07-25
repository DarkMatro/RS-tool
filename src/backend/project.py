"""
Control projects

classes:
    * Project
"""
from os import environ
from pathlib import Path
from shelve import open as shelve_open
from zipfile import ZipFile, ZIP_DEFLATED

from qtpy.QtWidgets import QMainWindow, QFileDialog

from qfluentwidgets import MessageBox
from ..data.config import get_config
from ..files.recent_files import RecentList


class Project:
    """
    Control projects, open, create new, close, save
    """

    def __init__(self, parent: QMainWindow) -> None:
        self.parent = parent
        self.recent_list = RecentList(size=int(environ["recent_limit"]))
        self.recent_menu = None
        self.project_path = ""

    def set_recent_menu(self, menu):
        self.recent_menu = menu
        self.update_recent_menu()

    def action_new_project(self) -> None:
        """
        Select file for new project.
        Save project to this file, clear all data in program.
        """
        if not self.can_close_project():
            return
        fd = QFileDialog(self.parent)
        file_path = fd.getSaveFileName(
            self.parent,
            "Create Project File",
            self.parent.latest_file_path,
            "ZIP (*.zip)",
        )
        if file_path[0] == "":
            return
        self.save_project(file_path[0])
        self.open_project(file_path[0])

    def action_open_project(self) -> None:
        if not self.can_close_project():
            return
        fd = QFileDialog(self.parent)
        file_path = fd.getOpenFileName(
            self.parent,
            "Select RS-tool project file to open",
            self.parent.latest_file_path,
            "(*.zip)",
        )
        if file_path[0] == "":
            return
        self.open_project(file_path[0])

    def action_open_recent(self, path):
        if Path(path).exists():
            if not self.can_close_project():
                return
            self.open_project(path)
        else:
            msg = MessageBox(
                "Recent file open error.", "Selected file doesn't exists", self.parent, {"Ok"}
            )
            msg.setInformativeText(path)
            msg.exec()

    def action_save_project(self):
        if self.project_path == "":
            fd = QFileDialog(self.parent)
            file_path = fd.getSaveFileName(self.parent, "Create Project File",
                                           self.parent.latest_file_path, "ZIP (*.zip)")
            if file_path[0] == "":
                return
            self.parent.latest_file_path = str(Path(file_path[0]).parent)
            self.update_project_title(file_path[0])

        self.save_project(self.project_path)

    def action_close_project(self) -> None:
        self.parent.setWindowTitle('')
        self.parent.ui.projectLabel.setText('')
        self.project_path = ''
        self.clear_all_parameters()
        self.parent.decide_vertical_scroll_bar_visible()

    def update_project_title(self, path: str) -> None:
        self.parent.ui.projectLabel.setText(path)
        self.parent.setWindowTitle(path)
        self.recent_list.appendleft(path)
        self.recent_list.save()
        self.update_recent_menu()
        self.project_path = path

    def open_project(self, path: str) -> None:
        self.parent.latest_file_path = str(Path(path).parent)
        self.update_project_title(path)
        self.clear_all_parameters()
        self.load_params(path)

    def open_demo_project(self) -> None:
        """
        1. Open Demo project
        Returns
        -------
        None
        """
        path = get_config()['help']['demo_project_path']
        self.open_project(path)
        self.load_params(path)

    def load_params(self, path: str) -> None:
        cfg = get_config("texty")["load_params"]
        self.parent.progress.open_progress(cfg)
        self.parent.setEnabled(False)
        self.parent.context.decomposition.graph_drawing.deselect_selected_line()
        self.unzip_project_file(path)
        self.parent.context.preprocessing.update_plot_item(
            self.parent.drag_widget.get_current_widget_name())
        self.parent.context.decomposition.switch_template()
        if (self.parent.ui.fit_params_table.model().rowCount() != 0
                and self.parent.ui.deconv_lines_table.model().rowCount() != 0):
            self.parent.context.decomposition.graph_drawing.draw_all_curves()
        self.parent.decide_vertical_scroll_bar_visible()
        self.parent.progress.time_start = None
        self.parent.setEnabled(True)
        self.parent.progress.close_progress(cfg, self.clear_all_parameters)

    def clear_all_parameters(self):
        self.parent.progress.executor_stop()
        self.parent.ui.input_table.model().clear_dataframe()
        self.parent.ui.dec_table.model().clear_dataframe()
        self.parent.context.reset()

        self.parent.ui.by_one_control_button.setChecked(False)
        self.parent.ui.by_group_control_button.setChecked(False)
        self.parent.ui.all_control_button.setChecked(True)
        self.parent.drag_widget.set_standard_order()

    def update_recent_menu(self):
        if len(self.recent_list) == 0:
            self.recent_menu.setDisabled(True)
            return
        self.recent_menu.setDisabled(False)
        self.recent_menu.clear()
        for p in self.recent_list:
            action = self.recent_menu.addAction(p)
            action.triggered.connect(
                lambda checked=None, path=p: self.action_open_recent(path)
            )

    def save_project(self, path: str, production_export: bool = False) -> None:
        cfg = get_config("texty")["save_project"]
        self.parent.progress.open_progress(cfg)
        filename = str(Path(path).parent) + "/" + str(Path(path).stem)
        self.shelve_file(filename, production_export)

        with ZipFile(filename + ".zip", "w", ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(filename + ".dat", "data.dat")
            zf.write(filename + ".dir", "data.dir")
            zf.write(filename + ".bak", "data.bak")
        Path(filename + ".dat").unlink()
        Path(filename + ".dir").unlink()
        Path(filename + ".bak").unlink()
        self.parent.ui.statusBar.showMessage(
            "File saved.   " + filename + ".zip", 10000
        )
        self.parent.progress.close_progress(cfg)
        self.parent.progress.time_start = None
        self.parent.context.set_modified(False)

    def unzip_project_file(self, path: str) -> None:
        with ZipFile(path) as archive:
            directory = Path(path).parent
            archive.extractall(directory)
            if not Path(str(directory) + "/data.dat").exists():
                return
            file_name = str(directory) + "/data"
            self.unshelve_project_file(file_name)
            Path(str(directory) + "/data.dat").unlink()
            Path(str(directory) + "/data.dir").unlink()
            Path(str(directory) + "/data.bak").unlink()

    def shelve_file(self, filename: str, production_export: bool = False):
        print('shelve_file', filename)
        with shelve_open(filename, "n") as db:
            db['widgets_order'] = self.parent.drag_widget.get_widgets_order()
            db["GroupsTable"] = self.parent.context.group_table.read()
            # TODO убирать data, оставить остальное
            if production_export:
                return
            db["InputTable"] = self.parent.ui.input_table.model().dataframe()
            db["InputData"] = self.parent.context.preprocessing.stages.input_data.read()
            db["ConvertData"] = self.parent.context.preprocessing.stages.convert_data.read()
            db["CutData"] = self.parent.context.preprocessing.stages.cut_data.read()
            db["NormalizedData"] = self.parent.context.preprocessing.stages.normalized_data.read()
            db["SmoothedData"] = self.parent.context.preprocessing.stages.smoothed_data.read()
            db["BaselineData"] = self.parent.context.preprocessing.stages.bl_data.read()
            db["TrimData"] = self.parent.context.preprocessing.stages.trim_data.read()
            db["AvData"] = self.parent.context.preprocessing.stages.av_data.read()
            db["Decomposition"] = self.parent.context.decomposition.read()
            db["Datasets"] = self.parent.context.datasets.read()
            db["ML"] = self.parent.context.ml.read()

    def unshelve_project_file(self, file_name: str) -> None:
        with shelve_open(file_name, "r") as db:
            if 'Decomposition' in db:
                self.parent.context.decomposition.load(db["Decomposition"])
            if "InputTable" in db:
                df = db["InputTable"]
                self.parent.ui.input_table.model().set_dataframe(df)
                self.parent.ui.dec_table.model().concat_deconv_table(filename=df.index)
            if "GroupsTable" in db:
                self.parent.context.group_table.load(db["GroupsTable"])
            if "InputData" in db:
                self.parent.context.preprocessing.stages.input_data.load(db["InputData"])
            if "ConvertData" in db:
                self.parent.context.preprocessing.stages.convert_data.load(db["ConvertData"])
            if 'CutData' in db:
                self.parent.context.preprocessing.stages.cut_data.load(db["CutData"])
            if 'NormalizedData' in db:
                self.parent.context.preprocessing.stages.normalized_data.load(db["NormalizedData"])
            if 'SmoothedData' in db:
                self.parent.context.preprocessing.stages.smoothed_data.load(db["SmoothedData"])
            if 'BaselineData' in db:
                self.parent.context.preprocessing.stages.bl_data.load(db["BaselineData"])
            if 'TrimData' in db:
                self.parent.context.preprocessing.stages.trim_data.load(db["TrimData"])
            if 'AvData' in db:
                self.parent.context.preprocessing.stages.av_data.load(db["AvData"])
            if 'widgets_order' in db:
                self.parent.drag_widget.set_order(db["widgets_order"])
            if 'Datasets' in db:
                self.parent.context.datasets.load(db["Datasets"])
            if 'ML' in db:
                self.parent.context.ml.load(db["ML"])


    def can_close_project(self) -> bool:
        if not self.parent.context.modified:
            return True
        msg = MessageBox(
            "You have unsaved changes.",
            "Save changes before exit?",
            self.parent,
            {"Yes", "No", "Cancel"},
        )
        if self.project_path:
            msg.setInformativeText(self.project_path)
        result = msg.exec()
        if result == 1:
            self.action_save_project()
            return True
        elif result == 0:
            return False
        elif result == 2:
            return True
