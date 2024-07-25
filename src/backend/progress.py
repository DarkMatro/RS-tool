"""
Handle progress bars and process.

This module contains the following classes:
    * Progress
"""
import dataclasses
from asyncio import gather
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from functools import partial
from logging import info
from multiprocessing import Manager
from os import environ
from typing import Callable, Any, ItemsView

import optuna
from asyncqtpy import asyncSlot
from qtpy.QtCore import QObject
from qtpy.QtWidgets import QMainWindow
from qtpy.QtWinExtras import QWinTaskbarButton

from qfluentwidgets import IndeterminateProgressBar, ProgressBar, StateToolTip
from src.data.config import get_config
from src.stages.ml.functions.hyperopt import optuna_opt


@dataclasses.dataclass
class Bars:
    progress_bar: None
    task_bar: None
    state_tooltip: None

class Progress(QObject):
    def __init__(self, parent: QMainWindow, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.time_start = None
        self.bars = Bars(progress_bar=None, task_bar=None, state_tooltip=None)
        self.ex = None
        self.break_event = None
        self.futures = []

    def open_progress(self, texty: dict, n_files: int = 0):
        self.time_start = datetime.now()
        self.parent.ui.statusBar.showMessage(texty["status_bar_open_msg"])
        self.close_progress_bar()
        self.open_progress_dialog(texty["state_tooltip_open_msg"], maximum=n_files)
        self.open_progress_bar(max_value=n_files)

    def close_progress(self, texty: dict | None = None, func: Callable | None = None) -> bool:
        self.close_progress_bar()
        if (
                (
                        self.bars.state_tooltip is not None
                        and self.bars.state_tooltip.wasCanceled()
                )
                or environ["CANCEL"] == "1"
        ) and texty is not None:
            self.parent.ui.statusBar.showMessage(texty["status_bar_cancel_msg"])
            if func is not None:
                func()
            return True
        return False

    @asyncSlot()
    async def run_in_executor(
            self, operation: str, func: Callable, iter_by: list, *args, **kwargs
    ) -> tuple[Any]:
        n_limit = kwargs.pop('n_limit') if 'n_limit' in kwargs else None
        n_files = kwargs.pop('n_files') if 'n_files' in kwargs else 0
        n_limit = get_config()["n_files_limit"][operation] if n_limit is None else n_limit
        self.ex = ThreadPoolExecutor() if n_files < n_limit \
            else ProcessPoolExecutor()
        with Manager() as manager:
            self.break_event = manager.Event()
            if 'break_event_by_user' in kwargs:
                kwargs['break_event_by_user'] = self.break_event
            with self.ex as ex:
                if isinstance(iter_by, (ItemsView, list)):
                    self.futures = [
                        self.parent.loop.run_in_executor(ex, partial(func, i, *args, **kwargs))
                        for i in iter_by
                    ]
                else:
                    self.futures = [
                        self.parent.loop.run_in_executor(ex, partial(func, *args, **kwargs))
                    ]
                for f in self.futures:
                    f.add_done_callback(self.progress_indicator)
                result = await gather(*self.futures)
        return result

    def done_callback(self, study, trial):
        print(self, study, trial)
        print(f"Trial {trial.number} finished with value: {trial.value}")
        print(f"Parameters: {trial.params}")
        print(f"Best trial so far: {study.best_trial.number} with value: {study.best_trial.value}")
        if self.bars.progress_bar is None:
            return
        current_value = self.bars.progress_bar.value() + 1
        self.bars.progress_bar.setValue(current_value)
        if self.bars.state_tooltip is not None:
            self.bars.state_tooltip.setValue(current_value)
        if self.bars.task_bar.progress() is not None:
            self.bars.task_bar.progress().setValue(current_value)
            self.bars.task_bar.progress().show()

    def progress_indicator(self, _=None) -> None:
        if self.bars.progress_bar is None:
            return
        current_value = self.bars.progress_bar.value() + 1
        self.bars.progress_bar.setValue(current_value)
        if self.bars.state_tooltip is not None:
            self.bars.state_tooltip.setValue(current_value)
        if self.bars.task_bar.progress() is not None:
            self.bars.task_bar.progress().setValue(current_value)
            self.bars.task_bar.progress().show()

    def open_progress_dialog(self, text: str, maximum: int = 0) -> None:
        environ["CANCEL"] = "0"
        if self.bars.state_tooltip is None:
            content = "Please wait patiently"
            self.bars.state_tooltip = StateToolTip(text, content, self.parent, maximum)
            x = self.parent.ui.centralwidget.width() // 2 - 120
            y = self.parent.ui.centralwidget.height() // 2 - 50
            self.bars.state_tooltip.move(x, y)
            self.bars.state_tooltip.closedSignal.connect(self.executor_stop)
            self.bars.state_tooltip.show()

    def open_progress_bar(self, min_value: int = 0, max_value: int = 0) -> None:
        if max_value == 0:
            self.bars.progress_bar = IndeterminateProgressBar(self.parent)
        else:
            self.bars.progress_bar = ProgressBar(self.parent)
            self.bars.progress_bar.setRange(min_value, max_value)
        self.parent.ui.statusBar.insertPermanentWidget(0, self.bars.progress_bar, 1)
        self.bars.task_bar = QWinTaskbarButton()
        self.bars.task_bar.progress().setRange(min_value, max_value)
        self.bars.task_bar.setWindow(self.parent.windowHandle())
        self.bars.task_bar.progress().show()

    def close_progress_bar(self):
        if self.bars.progress_bar is not None:
            self.parent.ui.statusBar.removeWidget(self.bars.progress_bar)
            del self.bars.progress_bar
            self.bars.progress_bar = None
        if self.bars.task_bar is not None and self.bars.task_bar.progress() is not None:
            self.bars.task_bar.progress().hide()
            self.bars.task_bar.progress().stop()
        if self.bars.state_tooltip is not None:
            text = (
                "Completed! 😆"
                if not self.bars.state_tooltip.wasCanceled()
                else "Canceled! 🥲"
            )
            self.bars.state_tooltip.setContent(text)
            self.bars.state_tooltip.setState(True)
            self.bars.state_tooltip = None
        self.time_start = None

    def executor_stop(self):
        if not self.ex or self.break_event is None:
            return
        for f in self.futures:
            if not f.done():
                f.cancel()
        try:
            self.break_event.set()
        except FileNotFoundError as ex:
            info("self.break_event.set() in executor_stop error", ex)
        environ["CANCEL"] = "1"
        self.ex.shutdown(cancel_futures=True, wait=False)
        self.close_progress_bar()
        self.parent.ui.statusBar.showMessage("Operation canceled by user")

    def cancelled_by_user(self) -> bool:
        """
        Cancel button was pressed by user?

        Returns
        -------
        out: bool
            True if Cancel button pressed
        """
        if ((self.bars.state_tooltip is not None and self.bars.state_tooltip.wasCanceled())
                or environ["CANCEL"] == "1"):
            self.close_progress_bar()
            self.ui.statusBar.showMessage("Cancelled by user.")
            info("Cancelled by user")
            return True
        return False
