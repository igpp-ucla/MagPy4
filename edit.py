
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

#import pyqtgraph as pg
import numpy as np
from FF_Time import FFTIME
from math import sin, cos, acos, fabs, pi

import functools
import time

class EditUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Edit')
        Frame.resize(630,500)
        self.axes = ['X','Y','Z']

        gridLayout = QtWidgets.QGridLayout(Frame)

        vectorFrame = QtWidgets.QGroupBox('Data Vectors')
        #vsFrame.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))

        self.vectorLayout = QtWidgets.QVBoxLayout(vectorFrame)

        ## init data vector combobox dropdowns
        #self.axisCombos = []
        ##self.vectorEdits = []
        ##self.vectorLabels = []
        #for i,ax in enumerate(self.axes):
        #    combo = QtGui.QComboBox()
        #    #edit = QtGui.QLineEdit(f'B{ax}')
        #    #combo.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
        #    combo.addItem('')
        #    for s in window.DATASTRINGS:
        #        if ax.lower() in s.lower():
        #            combo.addItem(s)

        #    self.axisCombos.append(combo)
        #    #label = QtGui.QLabel('X Matches')
        #    self.vectorGrid.addWidget(combo, 0, i, 1, 1)
        #    #vectorGrid.addWidget(edit, 0, i, 1, 1)
        #    #vectorGrid.addWidget(label, 1, i, 1, 1)

        #    #self.vectorEdits.append(edit)
        #    #self.vectorLabels.append(label)

        gridLayout.addWidget(vectorFrame, 0, 0, 1, 1)

        # matrix A setup
        self.R = []
        rFrame = QtWidgets.QGroupBox('Rotation Matrix')
        rLayout = QtWidgets.QGridLayout(rFrame)
        for y in range(3):
            row = []
            for x in range(3):
                edit = QtGui.QLineEdit()
                edit.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly)
                edit.setText('0.0')
                rLayout.addWidget(edit, y, x, 1, 1)
                row.append(edit)
            self.R.append(row)

        gridLayout.addWidget(rFrame, 0, 1, 1, 1)

        # axis rotation frame
        bFrame = QtWidgets.QGroupBox('Matrix Builders')

        extraButtons = QtWidgets.QHBoxLayout()
        bLayout = QtWidgets.QVBoxLayout(bFrame)
        self.loadIdentity = QtGui.QPushButton('Load Identity')
        self.loadIdentity.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.loadZeros = QtGui.QPushButton('Load Zeros')
        self.loadZeros.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        extraButtons.addWidget(self.loadIdentity)
        extraButtons.addWidget(self.loadZeros)
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
        #bLayout.addStretch()

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
        leftButtons = QtWidgets.QVBoxLayout()
        loadMat = QtWidgets.QPushButton('Load Matrix')
        self.removeRow = QtWidgets.QPushButton('Remove Matrix')
        leftButtons.addWidget(loadMat)
        leftButtons.addWidget(self.removeRow)
        leftButtons.addStretch()
        hBotLayout.addLayout(leftButtons,1)
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

        self.ui.loadIdentity.clicked.connect(functools.partial(self.setMatrix, self.ui.R, self.IDENTITY))
        self.ui.loadZeros.clicked.connect(functools.partial(self.setMatrix, self.ui.R, self.empty()))
        self.ui.apply.clicked.connect(self.apply)
        self.ui.removeRow.clicked.connect(self.removeHistory)

        # determine how many rows to start with
        self.axisCombos = []
        found = []
        maxLen = 0
        for kw in ['BX','BY','BZ']:
            row = []
            for dstr in self.window.DATASTRINGS:
                if kw.lower() in dstr.lower():
                    row.append(dstr)
            found.append(row)
            maxLen = max(maxLen, len(row))

        for i in range(maxLen):
            self.addAxisRow()

        self.setAxisCombosBlocked(True)

        for r,dstrs in enumerate(found):
            for c,dstr in enumerate(dstrs):
                combo = self.axisCombos[c][r]
                index = combo.findText(dstr)
                if index >= 0:
                    combo.setCurrentIndex(index)

        self.setAxisCombosBlocked(False)

        self.checkVectorRows()

        #self.updateVectorSelections()

        self.ui.history.currentRowChanged.connect(self.onHistoryChanged)
        self.historyMatrix = [] # selected matrix from histor
        self.history = []
        self.addHistory(self.IDENTITY, 'Identity')

        for i,gb in enumerate(self.ui.genButtons):
            gb.clicked.connect(functools.partial(self.axisRotGen, self.ui.axes[i]))

        self.setMatrix(self.ui.R, self.IDENTITY)

    def empty(self): # return an empty 2D list in 3x3 matrix form
        return [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]

    # returns a copy of array a
    def copy(self, a):
        return [[a[0][0],a[0][1],a[0][2]],
                [a[1][0],a[1][1],a[1][2]],
                [a[2][0],a[2][1],a[2][2]]]

    # matrix are used as keys in data table
    # DATADICT will be dict with dicts for each dstr
    def toString(self, m):
        return f'{m[0][0]}{m[0][1]}{m[0][2]}{m[1][0]}{m[1][1]}{m[1][2]}{m[2][0]}{m[2][1]}{m[2][2]}'

    def setAxisCombosBlocked(self, blocked):
        for row in self.axisCombos:
            for combo in row:
                combo.blockSignals(blocked)

    def onAxisComboChanged(self, r, c, text):
        print(f'{r} {c} {text}')
        self.checkVectorRows()
        self.updateVectorSelections()

    def checkVectorRows(self):
        # check for empty rows
        # partially empty rows
        emptyRows = []
        for r,row in enumerate(self.axisCombos):
            if (not row[0].currentText() or 
                not row[1].currentText() or
                not row[2].currentText()):
                emptyRows.append(r)

        numEmpty = len(emptyRows)
        if numEmpty == 0: # and len(self.axisCombos) < 4:
            self.addAxisRow()
        elif numEmpty > 1: # remove lastmost empty row
            index = emptyRows[-1]
            del self.axisCombos[index]
            layout = self.ui.vectorLayout.takeAt(index)
            while layout.count():
                layout.takeAt(0).widget().deleteLater()


    def addAxisRow(self):
        # init data vector combobox dropdowns
        r = len(self.axisCombos)
        row = []
        newLayout = QtWidgets.QHBoxLayout()
        for i,ax in enumerate(self.ui.axes):
            combo = QtGui.QComboBox()
            #edit = QtGui.QLineEdit(f'B{ax}')
            #combo.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
            combo.addItem('')
            for s in self.window.DATASTRINGS:
                if ax.lower() in s.lower():
                    combo.addItem(s)
            combo.currentTextChanged.connect(functools.partial(self.onAxisComboChanged, r, i))
            newLayout.addWidget(combo)
            row.append(combo)
        self.ui.vectorLayout.addLayout(newLayout)
        self.axisCombos.append(row)

    def updateVectorSelections(self):
        # top row should always have all options
        # second row should have all minus top rows
        # third row all minus 1st 2nd etc
        # dont add new row if no possible options left????
        # this is prob only place string lookup stuff needs to happen actually

        self.setAxisCombosBlocked(True)

        colStrs = [] # total option list for each column
        for i,ax in enumerate(self.ui.axes):
            col = []
            for dstr in self.window.DATASTRINGS:
                if ax.lower() in dstr.lower():
                    col.append(dstr)
            colStrs.append(col)

        # this way you have to uncheck option for it to become available to other rows
        for r,row in enumerate(self.axisCombos):
            for i,combo in enumerate(row):
                txt = combo.currentText()
                col = colStrs[i]
                if txt in col:
                    index = col.index(txt)
                    col[index] = [txt]
                    #col.remove(txt)
                    

        for r,row in enumerate(self.axisCombos):
            for i,combo in enumerate(row):
                txt = combo.currentText()
                combo.clear()
                combo.addItem('')

                for s in colStrs[i]:
                    if isinstance(s, list):
                        if txt == s[0]:
                            combo.addItem(txt)
                            combo.setCurrentIndex(combo.count()-1)
                    else:
                        combo.addItem(s)

                #if txt:
                #    combo.addItem(txt)
                #    combo.setCurrentIndex(1)
                #combo.addItems(colStrs[i])
                
        # this way top row always has all options and reduces each row down
        #for r,row in enumerate(self.axisCombos):
        #    for i,combo in enumerate(row):
        #        txt = combo.currentText()
        #        combo.clear()
        #        combo.addItems(colStrs[i])
        #        if txt and txt in colStrs[i]:
        #            combo.setCurrentIndex(combo.findText(txt))
        #            colStrs[i].remove(txt)
        #        else:
        #            combo.setCurrentIndex(0)

        self.setAxisCombosBlocked(False)

    # mat is 2D list of label/lineEdits
    # maybe add precision option or checkbox somewhere
    # so u can store whole precision internally but display abbreviated by default
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

    # manual matrix mult with my list format
    def mult(self, a, b):
        N = self.empty()
        for r in self.i:
            for c in self.i:
                for i in self.i:
                    N[r][c] += a[r][i] * b[i][c]
        return N

    # adds an entry to history matrix list and a list item at end of history ui
    def addHistory(self, mat, name):
        self.history.append(self.copy(mat))
        self.ui.history.addItem(QtWidgets.QListWidgetItem(f'{name}'))   
        self.ui.history.setCurrentRow(self.ui.history.count()-1)

    # removes selected history
    def removeHistory(self):
        curRow = self.ui.history.currentRow()
        if curRow == 0:
            print('cannot remove original data')
            return
        del self.history[curRow]
        self.ui.history.setCurrentRow(curRow-1) # change before take item otherwise onHistory gets called with wrong row
        self.ui.history.takeItem(curRow)

        #for i in range(3): # remove array from datadict data list
        #    dstr = self.axisCombos[i].currentText()
        #    del self.window.DATADICT[dstr][curRow]

    def onHistoryChanged(self, row):
        #print(f'CHANGED {row}')
        self.historyMatrix = self.history[row]
        self.setMatrix(self.ui.M, self.historyMatrix)
        #for i in range(3):
        #    dstr = self.axisCombos[i].currentText()
        #    self.window.DATAINDEX[dstr] = row

        self.window.replotData()

    # stacks up chosen x y z vector, multiplies by rotation matrix, then replot
    def apply(self):
        startTime = time.time()

        # todo: if these change need to clear all processed data out
        xstr = self.axisCombos[0].currentText()
        ystr = self.axisCombos[1].currentText()
        zstr = self.axisCombos[2].currentText()

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

        print(f'{time.time() - startTime}')