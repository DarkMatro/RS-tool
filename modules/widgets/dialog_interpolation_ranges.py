from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QComboBox, QPushButton, QFrame, QFormLayout


class DialogListBox(QDialog):
    def __init__(self, title: str, checked_ranges: list[int, int]) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowOpacity(0.9)
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(170, 100)
        self.ranges = QComboBox()
        for i in checked_ranges:
            self.ranges.addItem(str(i), i)
        button = QPushButton("OK", self)
        button.clicked.connect(self.ok_button_clicked)
        d_frame = QFrame(self)
        form_layout = QFormLayout(d_frame)
        form_layout.addRow(self.ranges)
        form_layout.addRow(button)

    def get_result(self) -> tuple[int, int]:
        return self.ranges.currentData()

    def ok_button_clicked(self) -> None:
        self.accept()


