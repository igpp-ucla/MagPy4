# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FilterDialog.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_FilterDialog(object):
    def setupUi(self, FilterDialog):
        FilterDialog.setObjectName("FilterDialog")
        FilterDialog.resize(470, 442)
        FilterDialog.setModal(True)
        self.buttonBox = QtWidgets.QDialogButtonBox(FilterDialog)
        self.buttonBox.setGeometry(QtCore.QRect(120, 400, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.windowTypeGroupBox = QtWidgets.QGroupBox(FilterDialog)
        self.windowTypeGroupBox.setGeometry(QtCore.QRect(10, 240, 450, 150))
        self.windowTypeGroupBox.setObjectName("windowTypeGroupBox")
        self.windowTypeComboBox = QtWidgets.QComboBox(self.windowTypeGroupBox)
        self.windowTypeComboBox.setGeometry(QtCore.QRect(10, 20, 120, 20))
        self.windowTypeComboBox.setObjectName("windowTypeComboBox")
        self.windowTypeComboBox.addItem("")
        self.windowTypeComboBox.addItem("")
        self.windowTypeComboBox.addItem("")
        self.windowTypeComboBox.addItem("")
        self.windowTypeComboBox.addItem("")
        self.windowTypeComboBox.addItem("")
        self.stopBandAttenuationUnit = QtWidgets.QLabel(self.windowTypeGroupBox)
        self.stopBandAttenuationUnit.setGeometry(QtCore.QRect(420, 50, 20, 20))
        self.stopBandAttenuationUnit.setObjectName("stopBandAttenuationUnit")
        self.rippleUnit = QtWidgets.QLabel(self.windowTypeGroupBox)
        self.rippleUnit.setGeometry(QtCore.QRect(420, 80, 20, 20))
        self.rippleUnit.setObjectName("rippleUnit")
        self.transitionWidthUnit = QtWidgets.QLabel(self.windowTypeGroupBox)
        self.transitionWidthUnit.setGeometry(QtCore.QRect(420, 110, 20, 20))
        self.transitionWidthUnit.setObjectName("transitionWidthUnit")
        self.transitionWidthText = QtWidgets.QLineEdit(self.windowTypeGroupBox)
        self.transitionWidthText.setGeometry(QtCore.QRect(290, 110, 120, 20))
        self.transitionWidthText.setObjectName("transitionWidthText")
        self.transitionWidthLabel = QtWidgets.QLabel(self.windowTypeGroupBox)
        self.transitionWidthLabel.setGeometry(QtCore.QRect(150, 110, 120, 20))
        self.transitionWidthLabel.setObjectName("transitionWidthLabel")
        self.rippleText = QtWidgets.QLineEdit(self.windowTypeGroupBox)
        self.rippleText.setGeometry(QtCore.QRect(290, 80, 120, 20))
        self.rippleText.setObjectName("rippleText")
        self.rippleLabel = QtWidgets.QLabel(self.windowTypeGroupBox)
        self.rippleLabel.setGeometry(QtCore.QRect(150, 80, 120, 20))
        self.rippleLabel.setObjectName("rippleLabel")
        self.stopBandAttenuationLabel = QtWidgets.QLabel(self.windowTypeGroupBox)
        self.stopBandAttenuationLabel.setGeometry(QtCore.QRect(150, 50, 120, 20))
        self.stopBandAttenuationLabel.setObjectName("stopBandAttenuationLabel")
        self.stopBandAttenuationText = QtWidgets.QLineEdit(self.windowTypeGroupBox)
        self.stopBandAttenuationText.setGeometry(QtCore.QRect(290, 50, 120, 20))
        self.stopBandAttenuationText.setObjectName("stopBandAttenuationText")
        self.alphaLabel = QtWidgets.QLabel(self.windowTypeGroupBox)
        self.alphaLabel.setGeometry(QtCore.QRect(150, 20, 120, 20))
        self.alphaLabel.setObjectName("alphaLabel")
        self.alphaText = QtWidgets.QLineEdit(self.windowTypeGroupBox)
        self.alphaText.setGeometry(QtCore.QRect(290, 20, 120, 20))
        self.alphaText.setObjectName("alphaText")
        self.filterTypeGroupBox = QtWidgets.QGroupBox(FilterDialog)
        self.filterTypeGroupBox.setGeometry(QtCore.QRect(10, 10, 451, 120))
        self.filterTypeGroupBox.setObjectName("filterTypeGroupBox")
        self.filterTypeComboBox = QtWidgets.QComboBox(self.filterTypeGroupBox)
        self.filterTypeComboBox.setGeometry(QtCore.QRect(10, 20, 120, 20))
        self.filterTypeComboBox.setObjectName("filterTypeComboBox")
        self.filterTypeComboBox.addItem("")
        self.filterTypeComboBox.addItem("")
        self.filterTypeComboBox.addItem("")
        self.filterTypeComboBox.addItem("")
        self.lowCutoffFreqUnit = QtWidgets.QLabel(self.filterTypeGroupBox)
        self.lowCutoffFreqUnit.setGeometry(QtCore.QRect(420, 20, 20, 20))
        self.lowCutoffFreqUnit.setObjectName("lowCutoffFreqUnit")
        self.highCutoffFreqUnit = QtWidgets.QLabel(self.filterTypeGroupBox)
        self.highCutoffFreqUnit.setGeometry(QtCore.QRect(420, 50, 20, 20))
        self.highCutoffFreqUnit.setObjectName("highCutoffFreqUnit")
        self.highCutoffFreqLabel = QtWidgets.QLabel(self.filterTypeGroupBox)
        self.highCutoffFreqLabel.setGeometry(QtCore.QRect(150, 50, 120, 20))
        self.highCutoffFreqLabel.setObjectName("highCutoffFreqLabel")
        self.highCutoffFreqText = QtWidgets.QLineEdit(self.filterTypeGroupBox)
        self.highCutoffFreqText.setGeometry(QtCore.QRect(290, 50, 120, 20))
        self.highCutoffFreqText.setObjectName("highCutoffFreqText")
        self.lowCutoffFreqLabel = QtWidgets.QLabel(self.filterTypeGroupBox)
        self.lowCutoffFreqLabel.setGeometry(QtCore.QRect(150, 20, 120, 20))
        self.lowCutoffFreqLabel.setObjectName("lowCutoffFreqLabel")
        self.lowCutoffFreqText = QtWidgets.QLineEdit(self.filterTypeGroupBox)
        self.lowCutoffFreqText.setGeometry(QtCore.QRect(290, 20, 120, 20))
        self.lowCutoffFreqText.setObjectName("lowCutoffFreqText")
        self.nyquistFreqLabel = QtWidgets.QLabel(self.filterTypeGroupBox)
        self.nyquistFreqLabel.setGeometry(QtCore.QRect(150, 80, 120, 20))
        self.nyquistFreqLabel.setObjectName("nyquistFreqLabel")
        self.nyquistFreqText = QtWidgets.QLabel(self.filterTypeGroupBox)
        self.nyquistFreqText.setGeometry(QtCore.QRect(290, 80, 120, 20))
        self.nyquistFreqText.setText("")
        self.nyquistFreqText.setIndent(3)
        self.nyquistFreqText.setObjectName("nyquistFreqText")
        self.nyquistFreqUnit = QtWidgets.QLabel(self.filterTypeGroupBox)
        self.nyquistFreqUnit.setGeometry(QtCore.QRect(420, 80, 20, 20))
        self.nyquistFreqUnit.setObjectName("nyquistFreqUnit")
        self.filterWidthGroupBox = QtWidgets.QGroupBox(FilterDialog)
        self.filterWidthGroupBox.setGeometry(QtCore.QRect(10, 140, 450, 90))
        self.filterWidthGroupBox.setObjectName("filterWidthGroupBox")
        self.filterWidthComboBox = QtWidgets.QComboBox(self.filterWidthGroupBox)
        self.filterWidthComboBox.setGeometry(QtCore.QRect(10, 20, 120, 20))
        self.filterWidthComboBox.setObjectName("filterWidthComboBox")
        self.filterWidthComboBox.addItem("")
        self.filterWidthComboBox.addItem("")
        self.numPointsLabel = QtWidgets.QLabel(self.filterWidthGroupBox)
        self.numPointsLabel.setGeometry(QtCore.QRect(150, 20, 120, 20))
        self.numPointsLabel.setObjectName("numPointsLabel")
        self.numPointsText = QtWidgets.QLineEdit(self.filterWidthGroupBox)
        self.numPointsText.setGeometry(QtCore.QRect(290, 20, 120, 20))
        self.numPointsText.setObjectName("numPointsText")
        self.timespanLabel = QtWidgets.QLabel(self.filterWidthGroupBox)
        self.timespanLabel.setGeometry(QtCore.QRect(150, 50, 120, 20))
        self.timespanLabel.setObjectName("timespanLabel")
        self.timespanText = QtWidgets.QLineEdit(self.filterWidthGroupBox)
        self.timespanText.setGeometry(QtCore.QRect(290, 50, 120, 20))
        self.timespanText.setObjectName("timespanText")
        self.timespanUnit = QtWidgets.QLabel(self.filterWidthGroupBox)
        self.timespanUnit.setGeometry(QtCore.QRect(420, 50, 20, 20))
        self.timespanUnit.setObjectName("timespanUnit")

        self.retranslateUi(FilterDialog)
        self.buttonBox.accepted.connect(FilterDialog.accept)
        self.buttonBox.rejected.connect(FilterDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(FilterDialog)
        FilterDialog.setTabOrder(self.filterTypeComboBox, self.lowCutoffFreqText)
        FilterDialog.setTabOrder(self.lowCutoffFreqText, self.highCutoffFreqText)
        FilterDialog.setTabOrder(self.highCutoffFreqText, self.filterWidthComboBox)
        FilterDialog.setTabOrder(self.filterWidthComboBox, self.numPointsText)
        FilterDialog.setTabOrder(self.numPointsText, self.timespanText)
        FilterDialog.setTabOrder(self.timespanText, self.windowTypeComboBox)
        FilterDialog.setTabOrder(self.windowTypeComboBox, self.alphaText)
        FilterDialog.setTabOrder(self.alphaText, self.stopBandAttenuationText)
        FilterDialog.setTabOrder(self.stopBandAttenuationText, self.rippleText)
        FilterDialog.setTabOrder(self.rippleText, self.transitionWidthText)

    def retranslateUi(self, FilterDialog):
        _translate = QtCore.QCoreApplication.translate
        FilterDialog.setWindowTitle(_translate("FilterDialog", "Filter"))
        self.windowTypeGroupBox.setTitle(_translate("FilterDialog", "Window Type"))
        self.windowTypeComboBox.setItemText(0, _translate("FilterDialog", "Rectangular"))
        self.windowTypeComboBox.setItemText(1, _translate("FilterDialog", "Triangular"))
        self.windowTypeComboBox.setItemText(2, _translate("FilterDialog", "Hamming"))
        self.windowTypeComboBox.setItemText(3, _translate("FilterDialog", "Hanning"))
        self.windowTypeComboBox.setItemText(4, _translate("FilterDialog", "Kaiser"))
        self.windowTypeComboBox.setItemText(5, _translate("FilterDialog", "Chebyshev"))
        self.stopBandAttenuationUnit.setText(_translate("FilterDialog", "dB"))
        self.rippleUnit.setText(_translate("FilterDialog", "dB"))
        self.transitionWidthUnit.setText(_translate("FilterDialog", "Hz"))
        self.transitionWidthLabel.setText(_translate("FilterDialog", "Transition Width:"))
        self.rippleLabel.setText(_translate("FilterDialog", "Ripple:"))
        self.stopBandAttenuationLabel.setText(_translate("FilterDialog", "Stop Band Attenuation:"))
        self.alphaLabel.setText(_translate("FilterDialog", "Alpha:"))
        self.filterTypeGroupBox.setTitle(_translate("FilterDialog", "Filter Type"))
        self.filterTypeComboBox.setItemText(0, _translate("FilterDialog", "Low Pass"))
        self.filterTypeComboBox.setItemText(1, _translate("FilterDialog", "High Pass"))
        self.filterTypeComboBox.setItemText(2, _translate("FilterDialog", "Band Pass"))
        self.filterTypeComboBox.setItemText(3, _translate("FilterDialog", "Band Stop"))
        self.lowCutoffFreqUnit.setText(_translate("FilterDialog", "Hz"))
        self.highCutoffFreqUnit.setText(_translate("FilterDialog", "Hz"))
        self.highCutoffFreqLabel.setText(_translate("FilterDialog", "High Cutoff Frequency:"))
        self.lowCutoffFreqLabel.setText(_translate("FilterDialog", "Low Cutoff Frequency:"))
        self.nyquistFreqLabel.setText(_translate("FilterDialog", "Nyquist frequency:"))
        self.nyquistFreqUnit.setText(_translate("FilterDialog", "Hz"))
        self.filterWidthGroupBox.setTitle(_translate("FilterDialog", "Filter Width"))
        self.filterWidthComboBox.setItemText(0, _translate("FilterDialog", "Points"))
        self.filterWidthComboBox.setItemText(1, _translate("FilterDialog", "Time"))
        self.numPointsLabel.setText(_translate("FilterDialog", "Number of Points:"))
        self.timespanLabel.setText(_translate("FilterDialog", "Timespan:"))
        self.timespanUnit.setText(_translate("FilterDialog", "s"))
