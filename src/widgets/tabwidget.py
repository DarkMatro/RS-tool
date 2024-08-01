"""
Module for custom Qt tab widgets.

This module provides custom implementations of Qt tab widgets:
- `HorizontalTabBar`: A custom QTabBar that paints tabs horizontally with centered text.
- `TabWidget`: A QTabWidget that uses the `HorizontalTabBar` for tab display.

Imports
-------
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTabBar, QStylePainter, QStyle, QStyleOptionTab, QTabWidget
"""
from qtpy.QtGui import QPaintEvent
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTabBar, QStylePainter, QStyle, QStyleOptionTab, QTabWidget


class HorizontalTabBar(QTabBar):
    """
    Custom QTabBar that paints tabs horizontally with centered text.

    This class overrides the default QTabBar to ensure that tab text is centered
    and the tab shape is properly drawn.

    Methods
    -------
    paintEvent(event: QPaintEvent) -> None
        Paints the tab bar, customizing the appearance of each tab.
    tabSizeHint(index: int) -> int
        Returns the size hint for the tab at the given index, ensuring width is not less than height.
    """
    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Paints the tab bar with custom appearance for each tab.

        Parameters
        ----------
        event : QPaintEvent
            The paint event to be handled. This is provided by Qt's event system.
        """
        painter = QStylePainter(self)
        option = QStyleOptionTab()
        for index in range(self.count()):
            self.initStyleOption(option, index)
            painter.drawControl(QStyle.ControlElement.CE_TabBarTabShape, option)
            painter.drawText(self.tabRect(index),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextDontClip,
                             self.tabText(index))

    def tabSizeHint(self, index: int) -> int:
        """
        Provides a size hint for the tab at the given index, ensuring the width
        is at least as large as the height.

        Parameters
        ----------
        index : int
            The index of the tab for which the size hint is requested.

        Returns
        -------
        int
            The suggested size for the tab, with width not less than height.
        """
        size = QTabBar.tabSizeHint(self, index)
        if size.width() < size.height():
            size.transpose()
        return size


class TabWidget(QTabWidget):
    """
    Custom QTabWidget that uses HorizontalTabBar for tab display.

    This class extends QTabWidget to use a custom HorizontalTabBar for tabbing functionality.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget for this TabWidget (default is None).
    """
    def __init__(self, parent=None) -> None:
        """
        Initializes the TabWidget instance and sets the custom HorizontalTabBar.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget for this TabWidget (default is None).
        """
        QTabWidget.__init__(self, parent)
        self.setTabBar(HorizontalTabBar())
