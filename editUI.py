
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from mth import Mth
from MagPy4UI import TimeEdit
import functools

class EditUI(object):

    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Edit')

        w = 600 if window.OS == 'windows' else 800
        Frame.resize(w,350)

        mainLayout = QtWidgets.QVBoxLayout(Frame)
        upperLayout = QtWidgets.QHBoxLayout()
        leftLayout = QtWidgets.QVBoxLayout()
        rightLayout = QtWidgets.QVBoxLayout()
        upperLayout.addLayout(leftLayout,1)
        upperLayout.addLayout(rightLayout,1)
        mainLayout.addLayout(upperLayout)

        # this part gets built dynamically
        vectorFrame = QtWidgets.QGroupBox('Data Vectors')
        vectorFrame.setToolTip('Select x y z vectors of data to be rotated by next matrix')
        self.vectorLayout = QtWidgets.QVBoxLayout(vectorFrame)
        leftLayout.addWidget(vectorFrame)

        # axis rotation frame
        builderFrame = QtWidgets.QGroupBox('Matrix Builders')

        builderLayout = QtWidgets.QVBoxLayout(builderFrame)

        self.manRotButton = QtGui.QPushButton('Custom Rotation')
        self.manRotButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.manRotButton.setToolTip('Enter a custom matrix, or build one with simple builders')
        builderLayout.addWidget(self.manRotButton)

        self.minVarButton = QtGui.QPushButton('Minimum Variance')
        self.minVarButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.minVarButton.setToolTip('Build a rotation matrix by selecting a minimum variance window')
        builderLayout.addWidget(self.minVarButton)

        leftLayout.addWidget(builderFrame)
        leftLayout.addStretch()
        
        # history
        matFrame = QtWidgets.QGroupBox('Matrix History')
        matLayout = QtWidgets.QVBoxLayout(matFrame)

        mGrid = QtWidgets.QGridLayout()
        self.M = [] # matrix that is displayed in history
        for y in range(3):
            row = []
            for x in range(3):
                label = QtGui.QLabel('0.0')
                #label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                #label.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
                mGrid.addWidget(label, y, x, 1, 1)
                row.append(label)
            self.M.append(row)

        matLayout.addLayout(mGrid)

        self.extraLabel = QtWidgets.QLabel('')
        self.extraLabel.setContentsMargins(0,10,0,0) # left top right bottom
        self.extraLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        matLayout.addWidget(self.extraLabel)

        rightLayout.addWidget(matFrame)

        histFrame = QtWidgets.QGroupBox()

        histLayout = QtWidgets.QHBoxLayout(histFrame)
        leftButtons = QtWidgets.QVBoxLayout()

        self.removeRow = QtWidgets.QPushButton('Remove Matrix')
        self.removeRow.setToolTip('Removes currently selected matrix from history')
        leftButtons.addWidget(self.removeRow)
        leftButtons.addStretch()
        histLayout.addLayout(leftButtons,1)
        self.history = QtWidgets.QListWidget()
        self.history.setEditTriggers(QtWidgets.QAbstractItemView.SelectedClicked)
        histLayout.addWidget(self.history,2)

        rightLayout.addWidget(histFrame)


    def makeHorizontalLine(self):
        horizontal = QtWidgets.QFrame()
        horizontal.setFrameShape(QtWidgets.QFrame.HLine)
        horizontal.setFrameShadow(QtWidgets.QFrame.Sunken)
        return horizontal
        

class ManRotUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Manual Rotation')
        Frame.resize(300,300)

        self.layout = QtWidgets.QVBoxLayout(Frame)

                # matrix A setup
        self.R = [] # current rotation matrix
        rFrame = QtWidgets.QGroupBox('Rotation Matrix')
        rLayout = QtWidgets.QGridLayout(rFrame)
        for y in range(3):
            row = []
            for x in range(3):
                edit = QtGui.QLineEdit()
                edit.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly) #i dont even know if this does anything
                edit.setText('0.0')
                rLayout.addWidget(edit, y, x, 1, 1)
                row.append(edit)
            self.R.append(row)

        self.layout.addWidget(rFrame)

        extraButtons = QtWidgets.QHBoxLayout()
        self.loadIdentity = QtGui.QPushButton('Load Identity')
        self.loadIdentity.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.loadZeros = QtGui.QPushButton('Load Zeros')
        self.loadZeros.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.loadCurrentEditMatrix = QtGui.QPushButton('Load Current Edit Matrix')
        self.loadCurrentEditMatrix.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        extraButtons.addWidget(self.loadIdentity)
        extraButtons.addWidget(self.loadZeros)
        extraButtons.addWidget(self.loadCurrentEditMatrix)
        extraButtons.addStretch()
        self.layout.addLayout(extraButtons)

        axFrame = QtWidgets.QGroupBox('By Axis Angle')
        axLayout = QtWidgets.QHBoxLayout(axFrame)
        angleLabel = QtGui.QLabel('Angle')
        angleLabel.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.axisAngle = QtGui.QDoubleSpinBox()
        self.axisAngle.setWrapping(True)
        self.axisAngle.setMaximum(360.0)
        self.axisAngle.setSuffix('\u00B0')
        self.axisAngle.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        axLayout.addWidget(angleLabel)
        axLayout.addWidget(self.axisAngle)
        self.genButtons = []
        for ax in Mth.AXES:
            gb = QtGui.QPushButton(f'{ax}')
            gb.setMinimumWidth(5)
            gb.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
            axLayout.addWidget(gb)
            self.genButtons.append(gb)

        self.layout.addWidget(axFrame)

        self.layout.addStretch()

        self.applyButton = QtWidgets.QPushButton('Apply')
        self.applyButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        self.layout.addWidget(self.applyButton)



class MinVarUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Minimum Variance')
        Frame.resize(300,150)

        self.layout = QtWidgets.QVBoxLayout(Frame)

        # setup xyz vector dropdowns
        vectorLayout = QtWidgets.QHBoxLayout()
        self.vector = []
        for i,ax in enumerate(Mth.AXES):
            v = QtWidgets.QComboBox()
            for dstr in window.DATASTRINGS:
                if ax.lower() in dstr.lower():
                    v.addItem(dstr)
            self.vector.append(v)
            vectorLayout.addWidget(v)
        vectorLayout.addStretch()

        self.layout.addLayout(vectorLayout)

        # setup datetime edits
        self.timeEdit = TimeEdit(QtGui.QFont("monospace", 10 if window.OS == 'windows' else 14))
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())

        self.layout.addWidget(self.timeEdit.start)
        self.layout.addWidget(self.timeEdit.end)

        self.eigenValsLabel = QtWidgets.QLabel('')
        self.layout.addWidget(self.eigenValsLabel)

        self.layout.addStretch()

        self.applyButton = QtWidgets.QPushButton('Apply')
        self.applyButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        self.layout.addWidget(self.applyButton)

