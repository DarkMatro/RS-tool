# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'src/ui/forms/settings.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(510, 200)
        Form.setMinimumSize(QtCore.QSize(510, 200))
        Form.setMaximumSize(QtCore.QSize(510, 200))
        font = QtGui.QFont()
        font.setFamily("Ableton Sans Light")
        Form.setFont(font)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mainSettingsFrame = QtWidgets.QFrame(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainSettingsFrame.sizePolicy().hasHeightForWidth())
        self.mainSettingsFrame.setSizePolicy(sizePolicy)
        self.mainSettingsFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainSettingsFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mainSettingsFrame.setObjectName("mainSettingsFrame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.mainSettingsFrame)
        self.horizontalLayout.setContentsMargins(8, 8, 8, 8)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = TabWidget(self.mainSettingsFrame)
        self.tabWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.West)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setIconSize(QtCore.QSize(1, 1))
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setUsesScrollButtons(True)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.interfaceTab = QtWidgets.QWidget()
        self.interfaceTab.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.interfaceTab.setObjectName("interfaceTab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.interfaceTab)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.settings1_frame = QtWidgets.QFrame(self.interfaceTab)
        self.settings1_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.settings1_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.settings1_frame.setObjectName("settings1_frame")
        self.formLayout = QtWidgets.QFormLayout(self.settings1_frame)
        self.formLayout.setContentsMargins(45, -1, -1, -1)
        self.formLayout.setHorizontalSpacing(86)
        self.formLayout.setObjectName("formLayout")
        self.Theme_label = QtWidgets.QLabel(self.settings1_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Theme_label.sizePolicy().hasHeightForWidth())
        self.Theme_label.setSizePolicy(sizePolicy)
        self.Theme_label.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setFamily("Ableton Sans Light")
        font.setPointSize(9)
        self.Theme_label.setFont(font)
        self.Theme_label.setObjectName("Theme_label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.Theme_label)
        self.ThemeComboBox_Bckgrnd = ComboBox(self.settings1_frame)
        self.ThemeComboBox_Bckgrnd.setObjectName("ThemeComboBox_Bckgrnd")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.ThemeComboBox_Bckgrnd)
        self.label = QtWidgets.QLabel(self.settings1_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QtCore.QSize(200, 16777215))
        font = QtGui.QFont()
        font.setFamily("Ableton Sans Light")
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.ThemeComboBox_Color = ComboBox(self.settings1_frame)
        self.ThemeComboBox_Color.setObjectName("ThemeComboBox_Color")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.ThemeComboBox_Color)
        self.axis_size_label = QtWidgets.QLabel(self.settings1_frame)
        self.axis_size_label.setMaximumSize(QtCore.QSize(200, 15))
        font = QtGui.QFont()
        font.setFamily("Ableton Sans Light")
        font.setPointSize(9)
        self.axis_size_label.setFont(font)
        self.axis_size_label.setObjectName("axis_size_label")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.axis_size_label)
        self.axis_font_size_spinBox = SpinBox(self.settings1_frame)
        self.axis_font_size_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.axis_font_size_spinBox.setMinimum(1)
        self.axis_font_size_spinBox.setProperty("value", 10)
        self.axis_font_size_spinBox.setObjectName("axis_font_size_spinBox")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.axis_font_size_spinBox)
        self.axis_label_font_size = QtWidgets.QLabel(self.settings1_frame)
        self.axis_label_font_size.setMaximumSize(QtCore.QSize(200, 20))
        font = QtGui.QFont()
        font.setFamily("Ableton Sans Light")
        font.setPointSize(9)
        self.axis_label_font_size.setFont(font)
        self.axis_label_font_size.setObjectName("axis_label_font_size")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.axis_label_font_size)
        self.axis_label_font_size_spinBox = SpinBox(self.settings1_frame)
        self.axis_label_font_size_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.axis_label_font_size_spinBox.setMinimum(1)
        self.axis_label_font_size_spinBox.setProperty("value", 14)
        self.axis_label_font_size_spinBox.setObjectName("axis_label_font_size_spinBox")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.axis_label_font_size_spinBox)
        self.verticalLayout_2.addWidget(self.settings1_frame)
        self.tabWidget.addTab(self.interfaceTab, "")
        self.Page2hidden = QtWidgets.QWidget()
        self.Page2hidden.setInputMethodHints(QtCore.Qt.ImhNone)
        self.Page2hidden.setObjectName("Page2hidden")
        self.tabWidget.addTab(self.Page2hidden, "")
        self.FileTab = QtWidgets.QWidget()
        self.FileTab.setObjectName("FileTab")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.FileTab)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.settings3_frame = QtWidgets.QFrame(self.FileTab)
        self.settings3_frame.setObjectName("settings3_frame")
        self.settings2_frame = QtWidgets.QGridLayout(self.settings3_frame)
        self.settings2_frame.setContentsMargins(92, 0, 0, 0)
        self.settings2_frame.setSpacing(0)
        self.settings2_frame.setObjectName("settings2_frame")
        self.recent_limit_label = QtWidgets.QLabel(self.settings3_frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.recent_limit_label.sizePolicy().hasHeightForWidth())
        self.recent_limit_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Ableton Sans Light")
        font.setPointSize(9)
        self.recent_limit_label.setFont(font)
        self.recent_limit_label.setObjectName("recent_limit_label")
        self.settings2_frame.addWidget(self.recent_limit_label, 0, 0, 1, 1)
        self.undo_limit_spinBox = SpinBox(self.settings3_frame)
        self.undo_limit_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.undo_limit_spinBox.setMinimum(1)
        self.undo_limit_spinBox.setProperty("value", 20)
        self.undo_limit_spinBox.setObjectName("undo_limit_spinBox")
        self.settings2_frame.addWidget(self.undo_limit_spinBox, 1, 1, 1, 1)
        self.undo_limit_label = QtWidgets.QLabel(self.settings3_frame)
        font = QtGui.QFont()
        font.setFamily("Ableton Sans Light")
        font.setPointSize(9)
        self.undo_limit_label.setFont(font)
        self.undo_limit_label.setObjectName("undo_limit_label")
        self.settings2_frame.addWidget(self.undo_limit_label, 1, 0, 1, 1)
        self.recent_limit_spinBox = SpinBox(self.settings3_frame)
        self.recent_limit_spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.recent_limit_spinBox.setMinimum(1)
        self.recent_limit_spinBox.setProperty("value", 10)
        self.recent_limit_spinBox.setObjectName("recent_limit_spinBox")
        self.settings2_frame.addWidget(self.recent_limit_spinBox, 0, 1, 1, 1)
        self.gridLayout_4.addWidget(self.settings3_frame, 0, 0, 1, 1)
        self.tabWidget.addTab(self.FileTab, "")
        self.Page4hidden = QtWidgets.QWidget()
        self.Page4hidden.setObjectName("Page4hidden")
        self.tabWidget.addTab(self.Page4hidden, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        self.verticalLayout.addWidget(self.mainSettingsFrame)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.Theme_label.setText(_translate("Form", "Theme background"))
        self.label.setText(_translate("Form", "Theme color"))
        self.axis_size_label.setText(_translate("Form", "Axis font-size"))
        self.axis_label_font_size.setText(_translate("Form", "Axis label font-size"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.interfaceTab), _translate("Form", "Look"))
        self.recent_limit_label.setText(_translate("Form", "Recent files limit"))
        self.undo_limit_label.setText(_translate("Form", "Undo limit"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.FileTab), _translate("Form", "File"))
from qfluentwidgets import ComboBox, SpinBox
from src.widgets.tabwidget import TabWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
