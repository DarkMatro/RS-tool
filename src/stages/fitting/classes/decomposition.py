from PyQt5.QtCore import QObject


class DecompositionStage(QObject):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
