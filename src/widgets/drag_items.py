"""
QPushButton objects aren't usually draggable, so to handle the mouse movements and initiate a drag
we need to implement a subclass.

This module contains the following classes:
    * DragButton
    * DragItem
    * DragWidget
    * DragTargetIndicator
"""
from os import environ
from typing import Any
from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import Qt, QMimeData, QObject
from qtpy.QtGui import QDrag, QMouseEvent, QPixmap
from qtpy.QtWidgets import QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout


class DragButton(QPushButton):
    """
    Draggable PushButton.
    """

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        """
        We implement a mouseMoveEvent which accepts the single e parameter of the event. We check to
        see if the left mouse button is pressed on this event -- as it would be when dragging -- and
        then initiate a drag. To start a drag, we create a QDrag object, passing in self to give us
        access later to the widget that was dragged. We also must pass in mime data. This is used
        for including information about what is dragged, particularly for passing data between
        applications. However, as here, it is fine to leave this empty.

        Qt's QDrag handler natively provides a mechanism for showing dragged objects which we can
        use. We can update our DragButton class to pass a pixmap image to QDrag and this will be
        displayed under the mouse pointer as the drag occurs. To show the widget, we just need to
        get a QPixmap of the widget we're dragging.

        Finally, we initiate a drag by calling drag.exec_(Qt.MoveAction). As with dialogs exec_()
        starts a new event loop, blocking the main loop until the drag is complete. The parameter
        Qt.MoveAction tells the drag handler what type of operation is happening, so it can show the
        appropriate icon tip to the user.

        Parameters
        -------
        e: QMouseEvent
        """
        if e.buttons() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            drag.setMimeData(mime)

            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)

            drag.exec_(Qt.MoveAction)


class DragItem(QWidget):
    """
    Draggable Item.

    Parameters
    -------
    draggable: bool, default = True
        allow drag & drop for this item

    Attributes
    -------
    focused: bool
        True if widget is currently selected

    """

    def __init__(self, draggable: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data = None
        self.draggable = draggable
        self.backend_instance = None
        self.focused = False

    def set_data(self, data: Any) -> None:
        """
        Parameters
        -------
        data: Any
        """
        self.data = data

    def set_backend_instance(self, instance: QObject) -> None:
        """
        Parameters
        -------
        instance: QObject
        """
        self.backend_instance = instance

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        """
        Parameters
        -------
        e: QMouseEvent
        """
        if e.buttons() == Qt.LeftButton and self.draggable:
            drag = QDrag(self)
            mime = QMimeData()
            drag.setMimeData(mime)

            # Render at x2 pixel ratio to avoid blur on Retina screens.
            pixmap = QPixmap(self.size().width() * 2, self.size().height() * 2)
            pixmap.setDevicePixelRatio(2)
            self.render(pixmap)
            drag.setPixmap(pixmap)

            drag.exec_(Qt.MoveAction)
            self.show()  # Show this widget again, if it's dropped outside.

    def focus_in(self) -> None:
        """
        Change style when selected.
        """
        if 'widget_selected' not in environ:
            return
        self.setStyleSheet("#widget_header{background-color: " + environ['widget_selected'] + ";}"
                           + "#label{color: " + environ['inversePlotText'] + "}")
        self.focused = True

    def focus_out(self) -> None:
        """
        Change style when deselected.
        """
        if 'widget_header' not in environ:
            return
        self.setStyleSheet("#widget_header{background-color: " + environ['widget_header'] + ";}")
        self.focused = False


class DragWidget(QWidget):
    """
    Generic list sorting handler.
    """

    orderChanged = pyqtSignal(list)
    doubleClickedWidget = pyqtSignal(QWidget)

    def __init__(self, orientation=Qt.Orientation.Vertical):
        super().__init__()
        self.setAcceptDrops(True)
        self.order = []
        # Store the orientation for drag checks later.
        self.orientation = orientation

        if self.orientation == Qt.Orientation.Vertical:
            self.w_layout = QVBoxLayout()
        else:
            self.w_layout = QHBoxLayout()

        # Add the drag target indicator. This is invisible by default,
        # we show it and move it around while the drag is active.
        self._drag_target_indicator = DragTargetIndicator()
        self.w_layout.addWidget(self._drag_target_indicator)
        self._drag_target_indicator.hide()
        self.setLayout(self.w_layout)

    def mouseDoubleClickEvent(self, _: QMouseEvent) -> None:
        """
        Emit widgetSelected signal for currently selected widget.

        Parameters
        -------
        _: QMouseEvent
        """
        for n in range(self.w_layout.count()):
            w = self.w_layout.itemAt(n).widget()
            if isinstance(w, DragTargetIndicator):
                continue
            if w.underMouse():
                self.doubleClickedWidget.emit(w)
                w.focus_in()
            else:
                w.focus_out()

    def dragEnterEvent(self, e: QMouseEvent) -> None:
        """
        Parameters
        -------
        e: QMouseEvent
        """
        e.accept()

    def dragLeaveEvent(self, e: QMouseEvent) -> None:
        """
        If you run the code at this point the drag behavior will work as expected. But if you drag
        the widget outside the window and drop you'll notice a problem: the target indicator will
        stay in place, but dropping the item won't drop the item in that position (the drop will be
         cancelled).

        To fix that we need to implement a dragLeaveEvent which hides the indicator.

        Parameters
        -------
        e: QMouseEvent
        """
        self._drag_target_indicator.hide()
        e.accept()

    def dragMoveEvent(self, e: QMouseEvent) -> None:
        """
        Modify the DragWidget.dragMoveEvent to show the drag target indicator. We show it by
        inserting it into the layout and then calling .show -- inserting a widget which is already
        in a layout will move it. We also hide the original item which is being dragged.

        Parameters
        -------
        e: QMouseEvent
        """
        source = e.source()
        if source is None:
            return
        # Find the correct location of the drop target, so we can move it there.
        index = self._find_drop_location(e)
        if index is not None:
            # Inserting moves the item if its already in the layout.
            self.w_layout.insertWidget(index, self._drag_target_indicator)
            # Hide the item being dragged.
            source.hide()
            # Show the target.
            self._drag_target_indicator.show()
        e.accept()

    def dropEvent(self, e: QMouseEvent) -> None:
        """
        Parameters
        -------
        e: QMouseEvent
        """
        widget = e.source()
        if widget is None:
            return
        # Use drop target location for destination, then remove it.
        self._drag_target_indicator.hide()
        index = self.w_layout.indexOf(self._drag_target_indicator)
        if index is not None:
            self.w_layout.insertWidget(index, widget)
            order = self.get_item_data()
            self.orderChanged.emit(order)
            self.order = order
            widget.show()
            self.w_layout.activate()
        e.accept()

    def _find_drop_location(self, e: QMouseEvent) -> int:
        """
        The method self._find_drop_location finds the index where the drag target will be shown
        (or the item dropped when the mouse released). We'll implement that next.

        The calculation of the drop location follows the same pattern as before. We iterate over
        the items in the layout and calculate whether our mouse drop location is to the left of
        each widget. If it isn't to the left of any widget, we drop on the far right.

        The drop location n is returned for use in the dragMoveEvent to place the drop target
        indicator.

        Parameters
        -------
        e: QMouseEvent

        Returns
        -------
        n: int
            drop location
        """
        pos = e.pos()
        spacing = self.w_layout.spacing() / 2
        n = 3
        for n in range(3, self.w_layout.count() - 1):
            # Get the widget at each index in turn.
            w = self.w_layout.itemAt(n).widget()

            if self.orientation == Qt.Orientation.Vertical:
                # Drag drop vertically.
                drop_here = w.y() - spacing <= pos.y() <= w.y() + w.size().height() + spacing
            else:
                # Drag drop horizontally.
                drop_here = w.x() - spacing <= pos.x() <= w.x() + w.size().width() + spacing

            if drop_here:
                # Drop over this target.
                break
        return n

    def add_item(self, item: QWidget):
        """
        Parameters
        -------
        item: QWidget
        """
        self.w_layout.addWidget(item)

    def get_item_data(self) -> list:
        """
        Returns
        -------
        data: list
        """
        data = []
        for n in range(self.w_layout.count()):
            # Get the widget at each index in turn.
            w = self.w_layout.itemAt(n).widget()
            if w != self._drag_target_indicator:
                # The target indicator has no data.
                data.append(w.data)
        return data

    def get_current_widget_name(self) -> str:
        """
        Returns name of backend class of currently selected widget.

        Returns
        -------
        class_name: str
            class name
        """
        class_name = ''
        for n in range(self.w_layout.count()):
            # Get the widget at each index in turn.
            w = self.w_layout.itemAt(n).widget()
            if isinstance(w, DragItem) and w.focused and w.backend_instance:
                # The target indicator has no data.
                class_name = w.backend_instance.name
        return class_name

    def get_current_widget(self) -> str:
        """
        Returns name of backend class of currently selected widget.

        Returns
        -------
        class_name: str
            class name
        """
        for n in range(self.w_layout.count()):
            # Get the widget at each index in turn.
            w = self.w_layout.itemAt(n).widget()
            if isinstance(w, DragItem) and w.focused and w.backend_instance:
                # The target indicator has no data.
                return w.backend_instance

    def get_previous_stage(self, current_stage: QObject) -> QObject | None:
        current_id = -1
        for i in range(self.w_layout.count() - 1, 0, -1):
            w = self.w_layout.itemAt(i).widget()
            if not isinstance(w, DragItem):
                continue
            if w.backend_instance == current_stage:
                current_id = i - 1
                break
        while current_id >= 0:
            w = self.w_layout.itemAt(current_id).widget()
            if isinstance(w, DragItem) and w.backend_instance.active:
                return w.backend_instance
            current_id -= 1
        return None

    def get_latest_active_stage(self):
        for i in range(self.w_layout.count() - 2, 0, -1):
            w = self.w_layout.itemAt(i).widget()
            if not isinstance(w, DragItem):
                continue
            if w.backend_instance.active:
                return w.backend_instance

    def get_next_stage(self, current_stage: QObject) -> QObject | None:
        ans = -1
        for i in range(self.w_layout.count() - 1, 0, -1):
            w = self.w_layout.itemAt(i).widget()
            if not isinstance(w, DragItem):
                continue
            if w.backend_instance == current_stage:
                ans = i + 1
                break
        return None if ans == -1 else self.w_layout.itemAt(ans).widget().backend_instance

    def get_widgets_order(self) -> list[str]:
        """
        Returns widget's names ordered.

        Returns
        -------
        order: list[str]
            names ordered as it is positioned
        """
        order = []
        for i in range(self.w_layout.count()):
            w = self.w_layout.itemAt(i).widget()
            if not isinstance(w, DragItem):
                continue
            order.append(w.backend_instance.name)
        return order

    def get_position_idx(self, class_name: str) -> int:
        idx = -1
        for i in range(self.w_layout.count()):
            w = self.w_layout.itemAt(i).widget()
            if not isinstance(w, DragItem):
                continue
            if w.backend_instance.name == class_name:
                return i
        return idx

    def get_widget_by_name(self, class_name: str) -> DragItem | None:
        for i in range(self.w_layout.count()):
            w = self.w_layout.itemAt(i).widget()
            if not isinstance(w, DragItem):
                continue
            if w.backend_instance.name == class_name:
                return w
        return

    def set_order(self, order: list[str]) -> None:
        """
        Move widgets corresponding to order.

        Parameters
        -------
        order: order: list[str]
            like ['InputData', 'ConvertData', 'CutData', 'NormalizedData', 'BaselineData',
             'SmoothedData', 'TrimData']
        """
        first_cut_data_idx = self.get_position_idx('CutData')
        idx = first_cut_data_idx + 1
        for i in range(3, len(order)):
            target_name = order[i]
            w = self.get_widget_by_name(target_name)
            if w is None:
                continue
            self.w_layout.removeWidget(w)
            self.w_layout.insertWidget(idx, w)
            idx += 1

    def set_standard_order(self):
        self.set_order(['InputData', 'ConvertData', 'CutData', 'BaselineData', 'SmoothedData',
                        'NormalizedData', 'TrimData'])


class DragTargetIndicator(QLabel):
    """
    The first step is to define our target indicator. This is just another label, which in our
    example is empty, with custom styles applied to make it have a solid "shadow" like background.
    This makes it obviously different to the items in the list, so it stands out as something
    distinct.
    If you change your list items, remember to also update the indicator dimensions to match.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(25, 5, 25, 5)
        self.setStyleSheet("QLabel { background-color: #ccc; border: 1px solid black; }")
