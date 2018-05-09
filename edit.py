
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

#import pyqtgraph as pg
import numpy as np
from FF_Time import FFTIME
from math import sin, cos, acos, fabs, pi

import functools

class EditUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Edit')
        Frame.resize(630,500)
        self.axes = ['X','Y','Z']

        gridLayout = QtWidgets.QGridLayout(Frame)

        vsFrame = QtWidgets.QGroupBox('Data Vector')
        #vsFrame.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        vsLayout = QtWidgets.QHBoxLayout(vsFrame)
        # init data vector combobox dropdowns
        self.axisCombos = []
        for ax in self.axes:
            combo = QtGui.QComboBox()
            #combo.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
            for s in window.DATASTRINGS:
                if ax.lower() in s.lower():
                    combo.addItem(s)
            vsLayout.addWidget(combo)
            self.axisCombos.append(combo)

        gridLayout.addWidget(vsFrame, 0, 0, 1, 1)


        # matrix A setup
        self.R = []
        rFrame = QtWidgets.QGroupBox('Rotation Matrix')
        rLayout = QtWidgets.QGridLayout(rFrame)
        for y in range(3):
            row = []
            for x in range(3):
                lineEdit = QtGui.QLineEdit()
                lineEdit.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly)
                lineEdit.setText('0.0')
                rLayout.addWidget(lineEdit, y, x, 1, 1)
                row.append(lineEdit)
            self.R.append(row)

        gridLayout.addWidget(rFrame, 0, 1, 1, 1)

        ## operator setup
        #opFrame = QtWidgets.QGroupBox('Operators')
        #self.opLayout = QtWidgets.QVBoxLayout(opFrame)
        #self.upload = QtGui.QPushButton('A => R')
        #self.operate = QtGui.QPushButton('AxR => R')
        #self.download = QtGui.QPushButton('A <= R')
        #self.opLayout.addWidget(self.upload)
        #self.opLayout.addWidget(self.operate)
        #self.opLayout.addWidget(self.download)

        ## matrix R setup
        #rotFrame = QtWidgets.QGroupBox('Rotation Matrix')
        #self.rgrid = QtWidgets.QGridLayout(rotFrame)
        #self.R = []
        #for y in range(3):
        #    row = []
        #    for x in range(3):
        #        label = QtGui.QLabel('0.0')
        #        self.rgrid.addWidget(label, y, x, 1, 1)
        #        row.append(label)
        #    self.R.append(row)

        #matrixLayout.addWidget(gridFrame,2)
        #matrixLayout.addWidget(opFrame,1)
        #matrixLayout.addWidget(rotFrame,2)
        #layout.addLayout(matrixLayout)

        #extraButtons = QtWidgets.QHBoxLayout()


        #extraButtons.addWidget(self.identity)
        #extraButtons.addStretch()
        
        #gridLayout.addLayout(extraButtons, 1,1,1,1)


        # axis rotation frame
        bFrame = QtWidgets.QGroupBox('Matrix Builders')

        extraButtons = QtWidgets.QHBoxLayout()
        bLayout = QtWidgets.QVBoxLayout(bFrame)
        self.identity = QtGui.QPushButton('Load Identity')
        self.identity.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        extraButtons.addWidget(self.identity)
        extraButtons.addStretch()
        bLayout.addLayout(extraButtons)

        axLayout = QtWidgets.QHBoxLayout()
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
        for ax in self.axes:
            gb = QtGui.QPushButton(f'{ax}')
            gb.setMinimumWidth(5)
            gb.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
            axLayout.addWidget(gb)
            self.genButtons.append(gb)

        bLayout.addLayout(axLayout)

        gridLayout.addWidget(bFrame, 2, 0, 1, 1)
        
        # history
        hFrame = QtWidgets.QGroupBox('History')
        hLayout = QtWidgets.QVBoxLayout(hFrame)

        mGrid = QtWidgets.QGridLayout()
        self.M = []
        for y in range(3):
            row = []
            for x in range(3):
                label = QtGui.QLabel('0.0')
                #label.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
                mGrid.addWidget(label, y, x, 1, 1)
                row.append(label)
            self.M.append(row)

        hLayout.addLayout(mGrid)

        hBotGrid = QtWidgets.QGroupBox()
        hBotLayout = QtWidgets.QHBoxLayout(hBotGrid)
        loadMatButton = QtWidgets.QPushButton('Load Matrix')
        hBotLayout.addWidget(loadMatButton,1)

        self.history = QtWidgets.QListWidget()
        #for i in range(5):
        #    item = QtWidgets.QListWidgetItem(f'list item {i}')
        #    self.history.insertItem(i,item)
        hBotLayout.addWidget(self.history,1)
        hLayout.addWidget(hBotGrid)

        gridLayout.addWidget(hFrame, 2, 1, 2, 1)


        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        gridLayout.addWidget(spacer, 4, 0, 1, 1)

        # bottom area with apply button
        bottomLayout = QtGui.QHBoxLayout()
        self.apply = QtGui.QPushButton('Apply')
        self.apply.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        bottomLayout.addWidget(self.apply)
        applyLabel = QtGui.QLabel('Multiplies data vector by rotation matrix')
        bottomLayout.addWidget(applyLabel)
        gridLayout.addLayout(bottomLayout, 5, 0, 2, 1)


class Edit(QtWidgets.QFrame, EditUI):

    def __init__(self, window, parent=None):
        super(Edit, self).__init__(parent)
        self.window = window
        self.ui = EditUI()
        self.ui.setupUI(self, window)

        self.i = [0, 1, 2]
        self.IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.FLAG = 1.e-10
        self.D2R = pi / 180.0 # degree to rad conversion constant

        #self.ui.upload.clicked.connect(self.upload)
        #self.ui.operate.clicked.connect(self.operate)
        #self.ui.download.clicked.connect(self.download)
        self.ui.identity.clicked.connect(functools.partial(self.setMatrix, self.ui.R, self.IDENTITY))
        self.ui.apply.clicked.connect(self.apply)

        self.ui.history.currentRowChanged.connect(self.onHistoryChanged)
        self.historyMatrix = [] # selected matrix from histor
        self.history = []
        self.addHistory(self.IDENTITY, 'Identity')
        #self.addHistory([[2,1,7],[5,5,-1],[20,100,2000]], 'Test Mat 1')
        #self.addHistory(self.genAxisRotationMatrix('Y', 90), 'Rot Matrix')
        #self.addHistory([[2000000000,10000000000,70000000000],[0,0,-30000000],[20,10000000,90000]], 'Test Mat 2')

        for i,gb in enumerate(self.ui.genButtons):
            gb.clicked.connect(functools.partial(self.axisRotGen, self.ui.axes[i]))

        self.setMatrix(self.ui.R, self.IDENTITY)

    def empty(self): # return new 2D list
        return [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]

    # returns a copy of array a
    def copy(self, a):
        return [[a[0][0],a[0][1],a[0][2]],
                [a[1][0],a[1][1],a[1][2]],
                [a[2][0],a[2][1],a[2][2]]]

    # mat is 2D list of label/lineEdits
    def getMatrix(self, mat):
        M = self.empty()
        for i in self.i:
            for j in self.i:
                s = mat[i][j].text()
                try:
                    f = float(s)
                except ValueError:
                    print(f'matrix has non-number at location {i},{j}')
                    f = 0.0
                M[i][j] = f

        #for r in M:
        #    print(f'[{r[0]}][{r[1]}][{r[2]}]')

        return M

    # mat is 2D list of label/lineEdits, m is 2D list of floats
    def setMatrix(self, mat, m):
        for i in self.i:
            for j in self.i:
                n = m[i][j]
                if fabs(n) > 9999.9999:
                    mat[i][j].setText(f'{n:.4e}')
                else:
                    mat[i][j].setText(f'{n:.4f}')

    def axisRotGen(self, axis):
        R = self.genAxisRotationMatrix(axis, self.ui.axisAngle.value())
        self.setMatrix(self.ui.R, R)

    # axis is x,y,z
    def genAxisRotationMatrix(self, axis, angle):
        angle *= self.D2R
        s = sin(angle)
        c = cos(angle)
        if fabs(s) < self.FLAG:
            s = 0
        if fabs(c) < self.FLAG:
            c = 0
        ax = axis.lower()
        if ax == 'x':
            return [[1, 0, 0], [0, c, s], [0, -s, c]]
        if ax == 'y':
            return [[c, 0, -s], [0, 1, 0], [s, 0, c]]
        if ax == 'z':
            return [[c, s, 0], [-s, c, 0], [0, 0, 1]]
        print(f'unknown axis "{ax}"')
        return self.copy(self.IDENTITY)

    #def upload(self):
    #    self.setMatrix(self.ui.R, self.getMatrix(self.ui.R))

    #def download(self):
    #    self.setMatrix(self.ui.R, self.getMatrix(self.ui.R))

    #def operate(self):
    #    A = self.getMatrix(self.ui.R)
    #    R = self.getMatrix(self.ui.R)
    #    N = self.empty()
    #    for r in self.i:
    #        for c in self.i:
    #            for i in self.i:
    #                N[r][c] += A[r][i] * R[i][c]
    #    self.setMatrix(self.ui.R, N)

    # manual matrix mult with my list format
    def mult(self, a, b):
        N = self.empty()
        for r in self.i:
            for c in self.i:
                for i in self.i:
                    N[r][c] += a[r][i] * b[i][c]
        return N

    def addHistory(self, mat, name):
        self.history.append(self.copy(mat))
        self.ui.history.addItem(QtWidgets.QListWidgetItem(f'{name}'))   
        self.ui.history.setCurrentRow(self.ui.history.count()-1)
        
    def onHistoryChanged(self, row):
        #print(f'CHANGED {row}')
        self.historyMatrix = self.history[row]
        self.setMatrix(self.ui.M, self.historyMatrix)
        for i in range(3):
            dstr = self.ui.axisCombos[i].currentText()
            self.window.DATAINDEX[dstr] = row

        self.window.replotData()

    # stacks up chosen x y z vector, multiplies by rotation matrix, then replot
    def apply(self):
        # todo: if these change need to clear all processed data out
        xstr = self.ui.axisCombos[0].currentText()
        ystr = self.ui.axisCombos[1].currentText()
        zstr = self.ui.axisCombos[2].currentText()

        X = self.window.DATADICT[xstr][0]
        Y = self.window.DATADICT[ystr][0]
        Z = self.window.DATADICT[zstr][0]

        # multiply by current selection in history
        R = self.mult(self.historyMatrix, self.getMatrix(self.ui.R))
        A = np.column_stack((X,Y,Z))
        M = np.matmul(A,R)

        self.window.DATADICT[xstr].append(M[:,0])
        self.window.DATADICT[ystr].append(M[:,1])
        self.window.DATADICT[zstr].append(M[:,2])

        self.addHistory(R, f'Matrix {len(self.history)}')