
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

        w = 600 if window.OS == 'windows' else 800
        Frame.resize(w,500)

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
        builderFrame = QtWidgets.QGroupBox('Matrix Builders')

        extraButtons = QtWidgets.QHBoxLayout()
        builderLayout = QtWidgets.QVBoxLayout(builderFrame)
        self.loadIdentity = QtGui.QPushButton('Load Identity')
        self.loadIdentity.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.loadZeros = QtGui.QPushButton('Load Zeros')
        self.loadZeros.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        extraButtons.addWidget(self.loadIdentity)
        extraButtons.addWidget(self.loadZeros)
        extraButtons.addStretch()
        builderLayout.addLayout(extraButtons)

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
        for ax in Edit.AXES:
            gb = QtGui.QPushButton(f'{ax}')
            gb.setMinimumWidth(5)
            gb.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
            axLayout.addWidget(gb)
            self.genButtons.append(gb)

        #builderLayout.addLayout(axLayout)
        builderLayout.addWidget(axFrame)
        #bLayout.addStretch()

        #mvFrame = QtWidgets.QGroupBox('Minimum Variance')
        #mvLayout = QtWidgets.QVBoxLayout(mvFrame)
        self.minVarButton = QtGui.QPushButton('Minimum Variance')
        self.minVarButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        #mvLayout.addWidget(self.minVarButton)
        #builderLayout.addWidget(mvFrame)
        builderLayout.addWidget(self.minVarButton)


        leftLayout.addWidget(builderFrame)
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
        self.history.setEditTriggers(QtWidgets.QAbstractItemView.SelectedClicked)
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
    AXES = ['X','Y','Z']
    i = [0, 1, 2]

    def empty(): # return an empty 2D list in 3x3 matrix form
        return [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
        # returns a copy of array a

    def copy(a):
        return [[a[0][0],a[0][1],a[0][2]],
                [a[1][0],a[1][1],a[1][2]],
                [a[2][0],a[2][1],a[2][2]]]

    # manual matrix mult with matrix list format
    def mult(a, b):
        N = Edit.empty()
        for r in Edit.i:
            for c in Edit.i:
                for i in Edit.i:
                    N[r][c] += a[r][i] * b[i][c]
        return N

    # converts float to string
    def formatNumber(num):
        n = round(num, Edit.STRING_PRECISION)
        #if n >= 10000 or n <= 0.0001: #not sure how to handle this for now
            #return f'{n:e}'
        return f'{n}'

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


    # ------------------------------------------------------------------
    # END OF STATICS
    # ------------------------------------------------------------------

    def __init__(self, window, parent=None):
        super(Edit, self).__init__(parent)
        self.window = window
        self.ui = EditUI()
        self.ui.setupUI(self, window)

        self.FLAG = 1.e-10
        self.D2R = pi / 180.0 # degree to rad conversion constant

        self.ui.loadIdentity.clicked.connect(functools.partial(self.setMatrix, self.ui.R, Edit.IDENTITY))
        self.ui.loadZeros.clicked.connect(functools.partial(self.setMatrix, self.ui.R, Edit.empty()))
        self.ui.apply.clicked.connect(self.apply)
        self.ui.removeRow.clicked.connect(self.removeHistory)

        # setup default BX vector rows
        self.axisDropdowns = []
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

        self.setVectorDropdownsBlocked(True)
        for col,dstrs in enumerate(found):
            for row,dstr in enumerate(dstrs):
                dd = self.axisDropdowns[row][col]
                index = dd.findText(dstr)
                if index >= 0:
                    dd.setCurrentIndex(index)
        self.setVectorDropdownsBlocked(False)

        # one run of this so dropdowns are correct
        self.checkVectorRows()
        self.updateVectorSelections()

        self.selectedMatrix = [] # selected matrix from history
        self.history = [] # list of matrices
        if self.window.editHistory:
            for mat,name in self.window.editHistory:
                if mat:
                    self.addHistory(mat, name)
                else:
                    self.ui.history.setCurrentRow(name)
        else:
            self.addHistory(Edit.IDENTITY, 'Identity')

        self.ui.history.currentRowChanged.connect(self.onHistoryChanged)
        self.onHistoryChanged(self.ui.history.currentRow())

        self.lastGeneratorAbbreviated = 'C'
        self.lastGeneratorName = 'Custom'

        for i,gb in enumerate(self.ui.genButtons):
            gb.clicked.connect(functools.partial(self.axisRotGen, Edit.AXES[i]))

        self.minVar = None
        self.ui.minVarButton.clicked.connect(self.openMinVar)

        self.setMatrix(self.ui.R, Edit.IDENTITY)

    def closeEvent(self, event):
        # save edit history
        hist = []
        for i in range(len(self.history)):
            hist.append((self.history[i],self.ui.history.item(i).text()))
            #print(f'{hist[i][0]} {hist[i][1]}' )
        hist.append(([], self.ui.history.currentRow())) # save row that was selected as last element

        self.window.editHistory = hist

        self.closeMinVar()

    def closeMinVar(self):
        if self.minVar:
            self.minVar.close()
            self.minVar = None

    def openMinVar(self):
        self.closeMinVar()
        self.minVar = MinVar(self, self.window)
        self.minVar.show()

    def setVectorDropdownsBlocked(self, blocked):
        for row in self.axisDropdowns:
            for dd in row:
                dd.blockSignals(blocked)

    def onAxisDropdownChanged(self, r, c, text):
        #print(f'{r} {c} {text}')
        self.checkVectorRows()
        self.updateVectorSelections()

    def checkVectorRows(self):
        # ensure at least one non full row
        # only get rid of empty rows

        nonFullRowCount = 0
        emptyRows = []
        for r,row in enumerate(self.axisDropdowns):
            r0 = not row[0].currentText()  
            r1 = not row[1].currentText()
            r2 = not row[2].currentText()
            if r0 or r1 or r2:
                nonFullRowCount += 1
            if r0 and r1 and r2:
                emptyRows.append(r)

        if nonFullRowCount == 0: # and len(self.axisDropdowns) < 4:
            self.addAxisRow()
        # if theres more than one non full row then delete empty rows
        elif nonFullRowCount > 1 and len(emptyRows) >= 1:
            index = emptyRows[-1]
            del self.axisDropdowns[index]
            layout = self.ui.vectorLayout.takeAt(index)
            while layout.count():
                layout.takeAt(0).widget().deleteLater()


    def addAxisRow(self):
        # init data vector dropdowns
        r = len(self.axisDropdowns)
        row = []
        newLayout = QtWidgets.QHBoxLayout()
        for i,ax in enumerate(Edit.AXES):
            dd = QtGui.QComboBox()
            dd.addItem('')
            for s in self.window.DATASTRINGS:
                if ax.lower() in s.lower():
                    dd.addItem(s)
            dd.currentTextChanged.connect(functools.partial(self.onAxisDropdownChanged, r, i))
            newLayout.addWidget(dd)
            row.append(dd)
        self.ui.vectorLayout.addLayout(newLayout)
        self.axisDropdowns.append(row)

    # sets up the vector axis dropdowns
    # if an item is checked it wont be available for the others in same column
    def updateVectorSelections(self):
        self.setVectorDropdownsBlocked(True) # block signals so we dont recursively explode

        colStrs = [] # total option list for each column
        for i,ax in enumerate(Edit.AXES):
            col = []
            for dstr in self.window.DATASTRINGS:
                if ax.lower() in dstr.lower():
                    col.append(dstr)
            colStrs.append(col)

        for r,row in enumerate(self.axisDropdowns):
            for i,dd in enumerate(row):
                txt = dd.currentText()
                col = colStrs[i]
                if txt in col:
                    # turn this entry into a list (done to keep the ordering correct)
                    # this is kinda stupid but just needed a simple way to mark this entry basically for next operation
                    col[col.index(txt)] = [txt] 

        for r,row in enumerate(self.axisDropdowns):
            for i,dd in enumerate(row):
                txt = dd.currentText()
                dd.clear()
                dd.addItem('') # empty option is always first
                for s in colStrs[i]:
                    # if its a list
                    if isinstance(s, list):
                        if txt == s[0]: # only add it and set to current if same as txt
                            dd.addItem(txt)
                            dd.setCurrentIndex(dd.count() - 1)
                    else: # add untaken options to list
                        dd.addItem(s)

        self.setVectorDropdownsBlocked(False)

    # mat is 2D list of label/lineEdits
    def getMatrix(self, mat):
        M = Edit.empty()
        for i in Edit.i:
            for j in Edit.i:
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
        for i in Edit.i:
            for j in Edit.i:
                mat[i][j].setText(Edit.formatNumber(m[i][j]))
                mat[i][j].repaint()

    def setRotationMatrix(self, m, name):
        self.setMatrix(self.ui.R, m)
        self.lastGeneratorName = name

    def axisRotGen(self, axis):
        angle = self.ui.axisAngle.value()
        R = self.genAxisRotationMatrix(axis, angle)
        self.setRotationMatrix(R, f'{axis}{angle}rot')

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
        return Edit.copy(Edit.IDENTITY)

    # adds an entry to history matrix list and a list item at end of history ui
    def addHistory(self, mat, name):
        self.history.append(Edit.copy(mat))

        # get names of items
        uihist = self.ui.history
        taken = set()
        for i in range(uihist.count()):
            taken.add(uihist.item(i).text())
            #print(uihist.item(i).text())
            #flags = uihist.item(i).flags()
            #flags |= QtCore.Qt.ItemIsEditable
            #uihist.item(i).setFlags(flags)
            pass

        newName = name
        fails = 1
        while newName in taken:
            newName = f'{name}({fails})'
            fails+=1
        name = newName

        item = QtWidgets.QListWidgetItem(f'{name}')
        flags = item.flags()
        flags |= QtCore.Qt.ItemIsEditable
        item.setFlags(flags)

        uihist.addItem(item)   
        uihist.setCurrentRow(uihist.count() - 1)

    # removes selected history
    def removeHistory(self):
        curRow = self.ui.history.currentRow()
        if curRow == 0:
            print('cannot remove original data')
            return
        del self.history[curRow]
        #self.ui.history.blockSignals(True)
        self.ui.history.setCurrentRow(curRow - 1) # change before take item otherwise onHistory gets called with wrong row
        self.ui.history.takeItem(curRow)
        # todo: check to see if memory consumption gets out of hand because not deleting data out of main window dictionaries ever

    def onHistoryChanged(self, row):
        #print(f'CHANGED {row}')
        self.selectedMatrix = self.history[row]
        self.setMatrix(self.ui.M, self.selectedMatrix)
        self.window.MATRIX = Edit.toString(self.selectedMatrix)
        self.window.replotData()

    def apply(self):
        R = Edit.mult(self.selectedMatrix, self.getMatrix(self.ui.R))
        self.generateData(R)
        self.addHistory(R, f'{self.lastGeneratorName}')

    # given current axis vector selections
    # make sure that all the correct data is calculated with matrix R
    def generateData(self, R):
        r = Edit.toString(R)
        i = Edit.identity()
        
        # for each full vector dropdown row 
        for dd in self.axisDropdowns:
            xstr = dd[0].currentText()
            ystr = dd[1].currentText()
            zstr = dd[2].currentText()

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


class MinVarUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Minimum Variance')
        Frame.resize(500,400)

        self.layout = QtWidgets.QVBoxLayout(Frame)

        # setup xyz vector dropdowns
        vectorLayout = QtWidgets.QHBoxLayout()
        self.vector = []
        for i,ax in enumerate(Edit.AXES):
            v = QtWidgets.QComboBox()
            for dstr in window.DATASTRINGS:
                if ax.lower() in dstr.lower():
                    v.addItem(dstr)
            self.vector.append(v)
            vectorLayout.addWidget(v)
        vectorLayout.addStretch()

        self.layout.addLayout(vectorLayout)

        # setup datetime edits
        sliderFont = QtGui.QFont("monospace", 10 if window.OS == 'windows' else 14)#, QtGui.QFont.Bold) 
        self.startTimeEdit = QtWidgets.QDateTimeEdit()
        self.endTimeEdit = QtWidgets.QDateTimeEdit()
        self.startTimeEdit.setFont(sliderFont)
        self.endTimeEdit.setFont(sliderFont)
        self.startTimeEdit.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.endTimeEdit.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.startTimeEdit.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")
        self.endTimeEdit.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")
        minDateTime,maxDateTime = window.getMinAndMaxDateTime()
        self.startTimeEdit.setMinimumDateTime(minDateTime)
        self.startTimeEdit.setMaximumDateTime(maxDateTime)
        self.startTimeEdit.setDateTime(minDateTime)
        self.endTimeEdit.setMinimumDateTime(minDateTime)
        self.endTimeEdit.setMaximumDateTime(maxDateTime)
        self.endTimeEdit.setDateTime(maxDateTime)
        self.layout.addWidget(self.startTimeEdit)
        self.layout.addWidget(self.endTimeEdit)

        self.eigenValsLabel = QtWidgets.QLabel('')
        self.layout.addWidget(self.eigenValsLabel)

        self.layout.addStretch()

        self.applyButton = QtWidgets.QPushButton('Apply')
        self.applyButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        self.layout.addWidget(self.applyButton)

class MinVar(QtWidgets.QFrame, MinVarUI):
    def __init__(self, edit, window, parent=None):
        super(MinVar, self).__init__(parent)#, QtCore.Qt.WindowStaysOnTopHint)
        self.edit = edit
        self.window = window

        self.ui = MinVarUI()
        self.ui.setupUI(self, window)

        self.window.startGeneralSelect('MINVAR', self.ui.startTimeEdit, self.ui.endTimeEdit)

        self.ui.applyButton.clicked.connect(self.calcMinVar)

        self.shouldResizeWindow = True

    def closeEvent(self, event):
        self.window.endGeneralSelect()

    def paintEvent(self, event):
        if self.shouldResizeWindow:
            self.shouldResizeWindow = False
            size = self.ui.layout.sizeHint()
            #size.x += 100
            self.resize(size)

    def average(self, vector):
        if vector is None:
            return 0
        size = vector.size
        if size == 0:
            return 0
        return np.sum(vector) / size

    def calcMinVar(self):
        # todo: test if need version of data where data gaps are removed and not smoothed, like makes array shorter
        # so you could select a length of data and it returns an array of only the valid values
        # otherwise minvar calcs prob get messed up when smoothed data probably affects the average

        iO,iE = self.window.getGeneralSelectTicks()

        xyz = []
        avg = []
        for v in self.ui.vector:
            data = self.window.getData(v.currentText())[iO:iE]
            xyz.append(data)
            avg.append(self.average(data))

        items = len(xyz[0])

        covar = Edit.empty()
        CoVar = Edit.empty()
        eigen = Edit.empty()

        for i in Edit.i:
            for j in Edit.i[i:]:
                for k in range(items):
                    covar[i][j] = covar[i][j] + xyz[i][k] * xyz[j][k]
        for i in Edit.i:
            for j in Edit.i[i:]:
                CoVar[i][j] = (covar[i][j] / items) - avg[i] * avg[j]
        # fill out lower triangle
        CoVar[1][0] = CoVar[0][1]
        CoVar[2][0] = CoVar[0][2]
        CoVar[2][1] = CoVar[1][2]

        A = np.array(CoVar)
        w, v = np.linalg.eigh(A, UPLO="U")

        for i in Edit.i:
            for j in Edit.i:
                eigen[2 - i][j] = v[j][i]
        eigenval = [w[2], w[1], w[0]]
        # force to be right handed
        e20 = eigen[0][1] * eigen[1][2] - eigen[0][2] * eigen[1][1]
        if e20 * eigen[2][0] < 0:
            eigen[2] = np.negative(eigen[2])
        if (eigen[0][2] < 0.):
            eigen[0] = np.negative(eigen[0])
            eigen[1] = np.negative(eigen[1])
        if (eigen[2][0] < 0.):
            eigen[1] = np.negative(eigen[1])
            eigen[2] = np.negative(eigen[2])
        
        self.edit.setRotationMatrix(eigen, 'minvar')
        print('min var calculation completed')
        print(eigenval)
