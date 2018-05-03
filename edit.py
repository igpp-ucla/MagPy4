
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

#import pyqtgraph as pg
import numpy as np
from FF_Time import FFTIME

class EditUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Edit')
        Frame.resize(630,500)

        layout = QtWidgets.QVBoxLayout(Frame)

        self.matrixPanel = QtGui.QWidget(Frame)

        # matrix A setup
        self.gridFrame = QtWidgets.QGroupBox('Matrix A',self.matrixPanel)
        self.gridFrame.setGeometry(QtCore.QRect(10,10,250,120))
        self.agrid = QtWidgets.QGridLayout(self.gridFrame)
        self.A = []
        for y in range(3):
            row = []
            for x in range(3):
                lineEdit = QtGui.QLineEdit()
                lineEdit.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly)
                lineEdit.setText('0.0')
                lineEdit.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
                self.agrid.addWidget(lineEdit, y, x, 1, 1)
                row.append(lineEdit)
            self.A.append(row)

        # operator setup
        self.opFrame = QtWidgets.QGroupBox('Operators',self.matrixPanel)
        self.opFrame.setGeometry(QtCore.QRect(265,10,80,120))
        self.opLayout = QtWidgets.QVBoxLayout(self.opFrame)
        self.upload = QtGui.QPushButton('A => R')
        self.operate = QtGui.QPushButton('AxR => R')
        self.download = QtGui.QPushButton('A <= R')
        self.opLayout.addWidget(self.upload)
        self.opLayout.addWidget(self.operate)
        self.opLayout.addWidget(self.download)

        # matrix R setup
        self.gridFrame = QtWidgets.QGroupBox('Rotation Matrix',self.matrixPanel)
        self.gridFrame.setGeometry(QtCore.QRect(350,10,250,120))
        self.rgrid = QtWidgets.QGridLayout(self.gridFrame)
        self.R = []
        for y in range(3):
            row = []
            for x in range(3):
                label = QtGui.QLabel()
                label.setText('0.0')
                label.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
                self.rgrid.addWidget(label, y, x, 1, 1)
                row.append(label)
            self.R.append(row)


        self.buttonBox = QtWidgets.QDialogButtonBox(Frame)
        self.buttonBox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Apply|QtGui.QDialogButtonBox.Reset)

        layout.addWidget(self.matrixPanel)
        layout.addWidget(self.buttonBox)

        # testing 
        self.A[0][2].setText('1.0')
        self.A[1][0].setText('2.0')

        #layout.addWidget(self.gridFrame)
        #layout.addWidget(self.opFrame)


class Edit(QtWidgets.QFrame, EditUI):
    def __init__(self, window, parent=None):
        super(Edit, self).__init__(parent)
        self.window = window
        self.ui = EditUI()
        self.ui.setupUI(self)
        
        #self.ui.updateButton.clicked.connect(self.updateSpectra)
        #self.ui.bandWidthSpinBox.valueChanged.connect(self.updateSpectra)
        #self.ui.oneTracePerCheckBox.stateChanged.connect(self.updateSpectra)
