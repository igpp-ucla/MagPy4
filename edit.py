
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

#import pyqtgraph as pg
import numpy as np
from FF_Time import FFTIME

import functools

class EditUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Edit')
        Frame.resize(630,500)

        layout = QtWidgets.QVBoxLayout(Frame)

        horizLayout = QtWidgets.QHBoxLayout()
        self.matrixPanel = QtGui.QWidget(Frame)

        # matrix A setup
        self.gridFrame = QtWidgets.QGroupBox('Matrix A')
        self.agrid = QtWidgets.QGridLayout(self.gridFrame)
        self.A = []
        for y in range(3):
            row = []
            for x in range(3):
                lineEdit = QtGui.QLineEdit()
                lineEdit.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly)
                lineEdit.setText('0.0')
                self.agrid.addWidget(lineEdit, y, x, 1, 1)
                row.append(lineEdit)
            self.A.append(row)

        # operator setup
        self.opFrame = QtWidgets.QGroupBox('Operators')
        self.opLayout = QtWidgets.QVBoxLayout(self.opFrame)
        self.upload = QtGui.QPushButton('A => R')
        self.operate = QtGui.QPushButton('AxR => R')
        self.download = QtGui.QPushButton('A <= R')
        self.opLayout.addWidget(self.upload)
        self.opLayout.addWidget(self.operate)
        self.opLayout.addWidget(self.download)

        # matrix R setup
        self.rotFrame = QtWidgets.QGroupBox('Rotation Matrix')
        self.rgrid = QtWidgets.QGridLayout(self.rotFrame)
        self.R = []
        for y in range(3):
            row = []
            for x in range(3):
                label = QtGui.QLabel()
                label.setText('0.0')
                self.rgrid.addWidget(label, y, x, 1, 1)
                row.append(label)
            self.R.append(row)

        horizLayout.addWidget(self.gridFrame,2)
        horizLayout.addWidget(self.opFrame,1)
        horizLayout.addWidget(self.rotFrame,2)


        layout.addLayout(horizLayout)

        extraButtons = QtWidgets.QHBoxLayout()

        self.identity = QtGui.QPushButton('Load Identity')
        self.identity.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        extraButtons.addWidget(self.identity)

        # init axis combobox dropdowns
        keywords = ['X','Y','Z']
        self.axisCombos = []
        for kw in keywords:
            combo = QtGui.QComboBox()
            for s in window.DATASTRINGS:
                if kw.lower() in s.lower():
                    combo.addItem(s)
            extraButtons.addWidget(combo)
            self.axisCombos.append(combo)

        layout.addLayout(extraButtons)

        layout.addStretch()
        self.apply = QtGui.QPushButton('Apply')
        self.apply.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        layout.addWidget(self.apply)

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
        self.ui.setupUI(self, window)

        self.i = [0, 1, 2]
        self.IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        self.ui.upload.clicked.connect(self.upload)
        self.ui.operate.clicked.connect(self.operate)
        self.ui.download.clicked.connect(self.download)
        self.ui.identity.clicked.connect(functools.partial(self.setMatrix, self.ui.A, self.IDENTITY))
        self.ui.apply.clicked.connect(self.apply)

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
                if n > 9999.9999:
                    mat[i][j].setText(f'{n:.4e}')
                else:
                    mat[i][j].setText(f'{n:.4f}')

    def upload(self):
        self.setMatrix(self.ui.R, self.getMatrix(self.ui.A))

    def download(self):
        self.setMatrix(self.ui.A, self.getMatrix(self.ui.R))

    def operate(self):
        A = self.getMatrix(self.ui.A)
        R = self.getMatrix(self.ui.R)
        N = self.empty()
        for r in self.i:
            for c in self.i:
                for i in self.i:
                    N[r][c] += A[r][i] * R[i][c]
        self.setMatrix(self.ui.R, N)

    def apply(self):
        import time
        startTime = time.time()

        xstr = self.ui.axisCombos[0].currentText()
        ystr = self.ui.axisCombos[1].currentText()
        zstr = self.ui.axisCombos[2].currentText()

        # use nogaps for now?
        X = self.window.DATADICTNOGAPS[xstr][-1]
        Y = self.window.DATADICTNOGAPS[ystr][-1]
        Z = self.window.DATADICTNOGAPS[zstr][-1]

        R = self.getMatrix(self.ui.R)
        A = np.column_stack((X,Y,Z))
        M = np.matmul(A,R)

        self.window.DATADICTNOGAPS[xstr].append(M[:,0])
        self.window.DATADICTNOGAPS[ystr].append(M[:,1])
        self.window.DATADICTNOGAPS[zstr].append(M[:,2])

        print(time.time() - startTime)

        self.window.replotData()