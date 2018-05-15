
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
        Frame.resize(600,500)
        self.axes = ['X','Y','Z']

        mainLayout = QtWidgets.QVBoxLayout(Frame)
        upperLayout = QtWidgets.QHBoxLayout()
        leftLayout = QtWidgets.QVBoxLayout()
        rightLayout = QtWidgets.QVBoxLayout()
        upperLayout.addLayout(leftLayout,1)
        upperLayout.addLayout(rightLayout,1)
        mainLayout.addLayout(upperLayout)

        # this part gets built dynamically
        vectorFrame = QtWidgets.QGroupBox('Data Vectors')
        self.vectorLayout = QtWidgets.QVBoxLayout(vectorFrame)
        leftLayout.addWidget(vectorFrame)

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

        rightLayout.addWidget(rFrame)

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

        leftLayout.addWidget(bFrame)
        leftLayout.addStretch()
        
        # history
        hFrame = QtWidgets.QGroupBox('Matrix History')
        hLayout = QtWidgets.QVBoxLayout(hFrame)

        mGrid = QtWidgets.QGridLayout()
        self.M = [] # matrix that is displayed in history
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
        #loadMat = QtWidgets.QPushButton('Load Matrix')
        self.removeRow = QtWidgets.QPushButton('Remove Matrix')
        #leftButtons.addWidget(loadMat)
        leftButtons.addWidget(self.removeRow)
        leftButtons.addStretch()
        hBotLayout.addLayout(leftButtons,1)
        self.history = QtWidgets.QListWidget()
        hBotLayout.addWidget(self.history,2)
        hLayout.addWidget(hBotGrid)

        rightLayout.addWidget(hFrame)

        mainLayout.addStretch()

        # bottom area with apply button
        bottomLayout = QtGui.QHBoxLayout()
        self.apply = QtGui.QPushButton('Apply')
        self.apply.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        bottomLayout.addWidget(self.apply)
        applyLabel = QtGui.QLabel('Multiplies each data vector by (selected history matrix multiplied by rotation matrix)')
        bottomLayout.addWidget(applyLabel)
        mainLayout.addLayout(bottomLayout)


class Edit(QtWidgets.QFrame, EditUI):

    IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    STRING_PRECISION = 10
    # i want to have it display and store at minimum precision necesary and go up to STRING_PRECISION
    # maybe also show as 1eX notation if over 10000? or somethin?

    def __init__(self, window, parent=None):
        super(Edit, self).__init__(parent)
        self.window = window
        self.ui = EditUI()
        self.ui.setupUI(self, window)

        self.i = [0, 1, 2]
        self.FLAG = 1.e-10
        self.D2R = pi / 180.0 # degree to rad conversion constant

        self.ui.loadIdentity.clicked.connect(functools.partial(self.setMatrix, self.ui.R, Edit.IDENTITY))
        self.ui.loadZeros.clicked.connect(functools.partial(self.setMatrix, self.ui.R, self.empty()))
        self.ui.apply.clicked.connect(self.apply)
        self.ui.removeRow.clicked.connect(self.removeHistory)

        # setup default BX vector rows
        self.axisCombos = []
        found = []
        maxLen = 0
        for kw in ['BX','BY','BZ']:
            f = []
            for dstr in self.window.DATASTRINGS:
                if kw.lower() in dstr.lower():
                    f.append(dstr)
            found.append(f)
            maxLen = max(maxLen, len(f))

        for i in range(maxLen):
            self.addAxisRow()

        self.setAxisCombosBlocked(True)
        for col,dstrs in enumerate(found):
            for row,dstr in enumerate(dstrs):
                combo = self.axisCombos[row][col]
                index = combo.findText(dstr)
                if index >= 0:
                    combo.setCurrentIndex(index)
        self.setAxisCombosBlocked(False)

        # one run of this so comboboxes are correct
        self.checkVectorRows()
        self.updateVectorSelections()

        self.ui.history.currentRowChanged.connect(self.onHistoryChanged)
        self.selectedMatrix = [] # selected matrix from history
        self.history = [] # list of matrices
        self.addHistory(Edit.IDENTITY, 'Identity')

        for i,gb in enumerate(self.ui.genButtons):
            gb.clicked.connect(functools.partial(self.axisRotGen, self.ui.axes[i]))

        self.setMatrix(self.ui.R, Edit.IDENTITY)

    def empty(self): # return an empty 2D list in 3x3 matrix form
        return [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]

    # returns a copy of array a
    def copy(self, a):
        return [[a[0][0],a[0][1],a[0][2]],
                [a[1][0],a[1][1],a[1][2]],
                [a[2][0],a[2][1],a[2][2]]]

    # matrix are stringified to use as keys in data table
    # DATADICT is dict with dicts for each dstr with (k, v) : (matrix str, modified data)
    # this is probably stupid but i dont know what im doing
    def toString(m, p=STRING_PRECISION):
        return (f'''[{Edit.formatNumber(m[0][0])}]
                    [{Edit.formatNumber(m[0][1])}]
                    [{Edit.formatNumber(m[0][2])}]
                    [{Edit.formatNumber(m[1][0])}]
                    [{Edit.formatNumber(m[1][1])}]
                    [{Edit.formatNumber(m[1][2])}]
                    [{Edit.formatNumber(m[2][0])}]
                    [{Edit.formatNumber(m[2][1])}]
                    [{Edit.formatNumber(m[2][2])}]''')

    def identity():
        return Edit.toString(Edit.IDENTITY)

    # converts float to string
    def formatNumber(num):
        n = round(num, Edit.STRING_PRECISION)
        #if n >= 10000 or n <= 0.0001: #not sure how to handle this for now
            #return f'{n:e}'
        return f'{n}'

    def setAxisCombosBlocked(self, blocked):
        for row in self.axisCombos:
            for combo in row:
                combo.blockSignals(blocked)

    def onAxisComboChanged(self, r, c, text):
        #print(f'{r} {c} {text}')
        self.checkVectorRows()
        self.updateVectorSelections()

    def checkVectorRows(self):
        # ensure at least one non full row
        # only get rid of empty rows

        nonFullRowCount = 0
        emptyRows = []
        for r,row in enumerate(self.axisCombos):
            r0 = not row[0].currentText()  
            r1 = not row[1].currentText()
            r2 = not row[2].currentText()
            if r0 or r1 or r2:
                nonFullRowCount += 1
            if r0 and r1 and r2:
                emptyRows.append(r)

        if nonFullRowCount == 0: # and len(self.axisCombos) < 4:
            self.addAxisRow()
        # if theres more than one non full row then delete empty rows
        elif nonFullRowCount > 1 and len(emptyRows) >= 1:
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

    # sets up the vector axis comboboxes
    # if an item is checke it wont be available for the others in same column
    def updateVectorSelections(self):
        self.setAxisCombosBlocked(True) # block signals so we dont recursively explode

        colStrs = [] # total option list for each column
        for i,ax in enumerate(self.ui.axes):
            col = []
            for dstr in self.window.DATASTRINGS:
                if ax.lower() in dstr.lower():
                    col.append(dstr)
            colStrs.append(col)

        for r,row in enumerate(self.axisCombos):
            for i,combo in enumerate(row):
                txt = combo.currentText()
                col = colStrs[i]
                if txt in col:
                    # turn this entry into a list (done to keep the ordering correct)
                    # this is kinda stupid but just needed a simple way to mark this entry basically for next operation
                    col[col.index(txt)] = [txt] 

        for r,row in enumerate(self.axisCombos):
            for i,combo in enumerate(row):
                txt = combo.currentText()
                combo.clear()
                combo.addItem('') # empty option is always first
                for s in colStrs[i]:
                    # if its a list
                    if isinstance(s, list):
                        if txt == s[0]: # only add it and set to current if same as txt
                            combo.addItem(txt)
                            combo.setCurrentIndex(combo.count() - 1)
                    else: # add untaken options to list
                        combo.addItem(s)
                combo.update()

        self.setAxisCombosBlocked(False)

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
        return M

    # mat is 2D list of label/lineEdits, m is 2D list of floats
    def setMatrix(self, mat, m):
        for i in self.i:
            for j in self.i:
                mat[i][j].setText(Edit.formatNumber(m[i][j]))

    def axisRotGen(self, axis):
        R = self.genAxisRotationMatrix(axis, self.ui.axisAngle.value())
        self.setMatrix(self.ui.R, R)

    # axis is x,y,z
    def genAxisRotationMatrix(self, axis, angle):
        angle *= self.D2R
        s = sin(angle)
        c = cos(angle)
        if fabs(s) < self.FLAG:
            s = 0.0
        if fabs(c) < self.FLAG:
            c = 0.0
        ax = axis.lower()
        if ax == 'x':
            return [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]]
        if ax == 'y':
            return [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
        if ax == 'z':
            return [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
        print(f'unknown axis "{ax}"')
        return self.copy(Edit.IDENTITY)

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
        self.ui.history.setCurrentRow(self.ui.history.count() - 1)

    # removes selected history
    def removeHistory(self):
        curRow = self.ui.history.currentRow()
        if curRow == 0:
            print('cannot remove original data')
            return
        del self.history[curRow]
        self.ui.history.setCurrentRow(curRow - 1) # change before take item otherwise onHistory gets called with wrong row
        self.ui.history.takeItem(curRow)
        # todo: check to see if memory consumption gets out of hand because not deleting anything currently

    def onHistoryChanged(self, row):
        #print(f'CHANGED {row}')
        self.selectedMatrix = self.history[row]
        self.setMatrix(self.ui.M, self.selectedMatrix)
        self.window.MATRIX = Edit.toString(self.selectedMatrix)
        self.window.replotData()

    def apply(self):
        R = self.mult(self.selectedMatrix, self.getMatrix(self.ui.R))
        self.generateData(R)
        self.addHistory(R, f'Matrix {len(self.history)}')

    # given current axis vector selections
    # make sure that all the correct data is calculated with matrix R
    def generateData(self, R):
        r = Edit.toString(R)
        i = Edit.identity()
        
        # for each full vector combo row 
        for combos in self.axisCombos:
            xstr = combos[0].currentText()
            ystr = combos[1].currentText()
            zstr = combos[2].currentText()

            if not xstr or not ystr or not zstr: # skip rows with empty selections
                continue

            # if datadict contains no entries for datastring with this matrix then generate
            if (r not in self.window.DATADICT[xstr] or
                r not in self.window.DATADICT[ystr] or
                r not in self.window.DATADICT[zstr]):

                # get original data
                X = self.window.DATADICT[xstr][i]
                Y = self.window.DATADICT[ystr][i]
                Z = self.window.DATADICT[zstr][i]

                A = np.column_stack((X,Y,Z))
                M = np.matmul(A,R)

                self.window.DATADICT[xstr][r] = M[:,0]
                self.window.DATADICT[ystr][r] = M[:,1]
                self.window.DATADICT[zstr][r] = M[:,2]
