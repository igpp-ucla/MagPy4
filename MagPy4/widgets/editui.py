
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from ..mth import Mth
from ..MagPy4UI import TimeEdit, MatrixWidget
import functools

class VectorLayout(QtWidgets.QGridLayout):
    def __init__(self, window):
        self.window = window
        self.dropdowns = []
        self.axisRows = []
        QtWidgets.QGridLayout.__init__(self)

    def flattenRows(self):
        return Mth.flattenLst(self.axisRows, 1)

    def setDefVecs(self, defVecs):
        self.axisRows = defVecs

    def getAxisVecs(self, row):
        if len(self.axisRows) == 0:
            return self.window.DATASTRINGS[:]
        elif row >= len(self.axisRows):
            return self.flattenRows()
        return self.axisRows[row]

    def addEmptyBox(self, row, col):
        box = QtWidgets.QComboBox()
        box.addItem('')
        self.addWidget(box, row, col)
        return box

    def buildDropdowns(self, vecsToUse):
        self.dropdowns = []
        for i in range(0, max(3, len(vecsToUse))):
            if i >= len(vecsToUse):
                initVecs = ['','','']
            else:
                initVecs = vecsToUse[i]
            ddRow = []
            for j in range(0, len(initVecs)):
                currVec = initVecs[j]
                box = self.addEmptyBox(i, j)
                restOfItems = list((set(self.getAxisVecs(i)) - set(initVecs)) | set([currVec]))
                if '' in restOfItems:
                    restOfItems.remove('')
                restOfItems.sort()
                box.addItems(restOfItems)
                box.setCurrentText(currVec)
                box.currentTextChanged.connect(self.rebuildDropdowns)
                ddRow.append(box)
            self.dropdowns.append(ddRow)

    def getDropdownVecs(self, ignore_empty=False):
        vecs = []
        for ddRow in self.dropdowns:
            rowLst = []
            for dd in ddRow:
                rowLst.append(dd.currentText())

            if ignore_empty and set(rowLst) == set(['']):
                continue

            vecs.append(rowLst)
        return vecs

    def rebuildDropdowns(self):
        prevVecs = self.getDropdownVecs()
        self.buildDropdowns(prevVecs)

class EditUI(object):

    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Edit')

        w = 600 if window.OS == 'windows' or window.OS == 'posix' else 800
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
        self.vecLt = VectorLayout(window)
        self.vectorLayout.addLayout(self.vecLt)

        leftLayout.addWidget(vectorFrame)

        # axis rotation frame
        builderFrame = QtWidgets.QGroupBox('Matrix Builders')
        builderLayout = QtWidgets.QVBoxLayout(builderFrame)

        self.customRotButton = QtWidgets.QPushButton('Custom Rotation...')
        self.customRotButton.setToolTip('Enter a custom matrix, or build one with simple builders')
        builderLayout.addWidget(self.customRotButton)

        self.minVarButton = QtWidgets.QPushButton('Minimum Variance...')
        self.minVarButton.setToolTip('Build a rotation matrix by selecting a minimum variance window')
        builderLayout.addWidget(self.minVarButton)

        for btn in [self.customRotButton, self.minVarButton]:
            btn.setFixedWidth(175)

        miscFrame = QtWidgets.QGroupBox('Other Edits')
        miscLayout = QtWidgets.QVBoxLayout(miscFrame)
        miscFrame.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum))

        # Filter button setup
        self.filterButton = QtWidgets.QPushButton('Filter...')
        self.filterButton.setToolTip('Apply various filters to smooth data')

        # Simple calculations setup
        self.calcBtn = QtWidgets.QPushButton('Calculate...')
        self.calcBtn.setToolTip('Perform simple calculations on data')

        # Smoothing tool setup
        self.smoothBtn = QtWidgets.QPushButton('Deglitch...')
        self.smoothBtn.setToolTip('Smooth and shift glitches in Insight data')

        btns = [self.filterButton, self.calcBtn]

        # Data removal tool setup
        self.dataFlagBtn = QtWidgets.QPushButton('Remove Data...')
        self.dataFlagBtn.setToolTip('Remove bad data or replace with flag values')
        if len(window.FIDs) == 1:
            btns += [self.dataFlagBtn]

        # GSM/GSE rotations
        self.gseGsmBtn = QtWidgets.QPushButton('GSE/GSM Coordinates...')
        btns += [self.gseGsmBtn]
        
        # Add buttons to layout
        for btn in btns:
            miscLayout.addWidget(btn)
            btn.setFixedWidth(175)

        leftLayout.addWidget(builderFrame)
        leftLayout.addWidget(miscFrame)
        
        # history
        matFrame = QtWidgets.QGroupBox('Matrix History')
        matLayout = QtWidgets.QVBoxLayout(matFrame)
        self.M = MatrixWidget()
        matLayout.addWidget(self.M)

        self.extraLabel = QtWidgets.QLabel('')
        self.extraLabel.setContentsMargins(0,10,0,0) # left top right bottom
        self.extraLabel.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        matLayout.addWidget(self.extraLabel)

        rightLayout.addWidget(matFrame)

        histFrame = QtWidgets.QGroupBox()

        histLayout = QtWidgets.QHBoxLayout(histFrame)
        leftButtons = QtWidgets.QVBoxLayout()

        self.removeRow = QtWidgets.QPushButton('Remove')
        self.removeRow.setToolTip('Removes currently selected matrix from history')
        leftButtons.addWidget(self.removeRow)
        leftButtons.addStretch()
        histLayout.addLayout(leftButtons,1)
        self.history = QtWidgets.QListWidget()
        histLayout.addWidget(self.history,2)

        rightLayout.addWidget(histFrame)

    def makeHorizontalLine(self):
        horizontal = QtWidgets.QFrame()
        horizontal.setFrameShape(QtWidgets.QFrame.HLine)
        horizontal.setFrameShadow(QtWidgets.QFrame.Sunken)
        return horizontal

class CustomRotUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Custom Rotation')
        Frame.resize(300,300)

        self.layout = QtWidgets.QVBoxLayout(Frame)
        self.layout.setSpacing(12)

        rFrame = QtWidgets.QGroupBox('Rotation Matrix')
        rLayout = QtWidgets.QVBoxLayout(rFrame)
        self.R = MatrixWidget(type='lines')
        rLayout.addWidget(self.R)
        self.layout.addWidget(rFrame)

        extraButtons = QtWidgets.QHBoxLayout()
        self.loadIdentity = QtWidgets.QPushButton('Load Identity')
        self.loadIdentity.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.loadZeros = QtWidgets.QPushButton('Load Zeros')
        self.loadZeros.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.loadCurrentEditMatrix = QtWidgets.QPushButton('Load Current Edit Matrix')
        self.loadCurrentEditMatrix.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        extraButtons.addWidget(self.loadIdentity)
        extraButtons.addWidget(self.loadZeros)
        extraButtons.addWidget(self.loadCurrentEditMatrix)

        # since no more generate button should reset when other buttons are pressed
        self.loadIdentity.clicked.connect(self.clearAxisAngle)
        self.loadZeros.clicked.connect(self.clearAxisAngle)
        self.loadCurrentEditMatrix.clicked.connect(self.clearAxisAngle)

        extraButtons.addStretch()
        self.layout.addLayout(extraButtons)

        axFrame = QtWidgets.QGroupBox('By Axis Angle')
        axLayout = QtWidgets.QHBoxLayout(axFrame)
        angleLabel = QtWidgets.QLabel('Angle:')
        angleLabel.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.axisAngle = QtWidgets.QDoubleSpinBox()
        self.axisAngle.setWrapping(True)
        self.axisAngle.setMaximum(360.0)
        self.axisAngle.setSuffix('\u00B0')
        self.axisAngle.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        axLayout.addWidget(angleLabel)
        axLayout.addWidget(self.axisAngle)
        self.genButtons = []
        for ax in Mth.AXES:
            gb = QtWidgets.QRadioButton(f'{ax}')
            gb.setMinimumWidth(5)
            gb.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
            axLayout.addWidget(gb)
            self.genButtons.append(gb)
        self.genButtons[0].setChecked(True)

        self.layout.addWidget(axFrame)

        insightFrame = QtWidgets.QGroupBox('Load Insight coordinate transformation matrices:')
        insightLt = QtWidgets.QHBoxLayout(insightFrame)
        self.spaceToLocBtn = QtWidgets.QPushButton('Spacecraft to Local Level')
        self.instrToSpaceBtn = QtWidgets.QPushButton('Instrument to Spacecraft')
        insightLt.addWidget(self.instrToSpaceBtn)
        insightLt.addWidget(self.spaceToLocBtn)
        if window.insightMode:
            self.layout.addWidget(insightFrame)

        self.layout.addStretch()

        self.applyButton = QtWidgets.QPushButton('Apply')
        self.applyButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        self.layout.addWidget(self.applyButton)

        self.layout.setAlignment(self.applyButton, QtCore.Qt.AlignRight)

    def clearAxisAngle(self):
        self.axisAngle.setValue(0.0)

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
            addedItems = []
            for dstr in window.DATASTRINGS:
                if ax.lower() in dstr.lower():
                    v.addItem(dstr)
                    addedItems.append(v)
            if addedItems == []:
                v.addItems(window.DATASTRINGS)
                v.setCurrentText(window.DATASTRINGS[i])
            self.vector.append(v)
            vectorLayout.addWidget(v)
        vectorLayout.addStretch()

        self.layout.addLayout(vectorLayout)

        # setup datetime edits
        self.timeEdit = TimeEdit()
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())

        # default it to the current time selection
        cmin,cmax = window.getCurrentDateTime()
        self.timeEdit.setStartNoCallback(cmin)
        self.timeEdit.setEndNoCallback(cmax)        

        self.layout.addWidget(self.timeEdit.start)
        self.layout.addWidget(self.timeEdit.end)

        self.eigenValsLabel = QtWidgets.QLabel('')
        self.layout.addWidget(self.eigenValsLabel)

        self.layout.addStretch()

        self.applyButton = QtWidgets.QPushButton('Apply')
        self.applyButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        self.layout.addWidget(self.applyButton)