
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

#import pyqtgraph as pg
import numpy as np
from FF_Time import FFTIME
from math import sin, cos, acos, fabs, pi

import functools
import time

from editUI import EditUI, ManRotUI, MinVarUI

from mth import Mth
from MagPy4UI import PyQtUtils

class Edit(QtWidgets.QFrame, EditUI):

    def moveToFront(window):
        if window:
            # this will remove minimized status 
            # and restore window with keeping maximized/normal state
            window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
            # this will activate the window
            window.activateWindow()


    def __init__(self, window, parent=None):
        super(Edit, self).__init__(parent)
        self.window = window
        self.ui = EditUI()
        self.ui.setupUI(self, window)

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
        self.history = [] # list tuples of matrices and string for extra data (ie eigenvalues)
        if self.window.editHistory:
            for mat,name in self.window.editHistory:
                if mat:
                    self.addHistory(mat[0], mat[1], name)
                else:
                    self.ui.history.setCurrentRow(name)
        else:
            self.addHistory(Mth.IDENTITY, 'original data', 'Identity')

        self.ui.history.currentRowChanged.connect(self.onHistoryChanged)
        self.onHistoryChanged(self.ui.history.currentRow())

        self.lastGeneratorAbbreviated = 'C'
        self.lastGeneratorName = 'Custom'

        self.minVar = None
        self.ui.minVarButton.clicked.connect(self.openMinVar)
        self.manRot = None
        self.ui.manRotButton.clicked.connect(self.openManRot)


    def closeEvent(self, event):
        # save edit history
        hist = []
        for i in range(len(self.history)):
            hist.append((self.history[i],self.ui.history.item(i).text()))
            #print(f'{hist[i][0]} {hist[i][1]}' )
        hist.append(([], self.ui.history.currentRow())) # save row that was selected as last element

        self.window.editHistory = hist

        self.closeManRot()
        self.closeMinVar()

    def openManRot(self):
        self.closeManRot()
        self.manRot = ManRot(self, self.window)
        self.manRot.show()
    def closeManRot(self):
        if self.manRot:
            self.manRot.close()
            self.manRot = None

    def openMinVar(self):
        self.closeMinVar()
        self.minVar = MinVar(self, self.window)
        self.minVar.show()
    def closeMinVar(self):
        if self.minVar:
            self.minVar.close()
            self.minVar = None


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
            PyQtUtils.clearLayout(layout)

    def addAxisRow(self):
        # init data vector dropdowns
        r = len(self.axisDropdowns)
        row = []
        newLayout = QtWidgets.QHBoxLayout()
        for i,ax in enumerate(Mth.AXES):
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
        for i,ax in enumerate(Mth.AXES):
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

    #def setRotationMatrix(self, m, name):
    #    Mth.setMatrix(self.ui.R, m)
    #    self.lastGeneratorName = name

    # adds an entry to history matrix list and a list item at end of history ui
    def addHistory(self, mat, extra, name):
        self.history.append((Mth.copy(mat),extra))

        # get names of items
        uihist = self.ui.history
        taken = set()
        for i in range(uihist.count()):
            taken.add(uihist.item(i).text())

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
        hist = self.history[row]
        self.selectedMatrix = hist[0]
        self.ui.extraLabel.setText(hist[1])
        Mth.setMatrix(self.ui.M, self.selectedMatrix)
        self.window.MATRIX = Mth.matToString(self.selectedMatrix)
        self.updateLabelNamesByMatrix(self.window.MATRIX, self.ui.history.item(row).text())
        self.window.replotData()

    def apply(self, mat, extra, name):
        R = Mth.mult(self.selectedMatrix, mat)
        self.generateData(R, '*')
        self.addHistory(R, extra, f'{name}')

    # matrix needs to be in string form
    def updateLabelNamesByMatrix(self, mat, name):
        isIdentity = mat == Mth.identity()
        for dstr in self.window.DATASTRINGS:
            datas = self.window.DATADICT[dstr]
            if mat in datas:
                datas[mat][1] = dstr if isIdentity else self.getEditedName(dstr, datas[mat][2], name)

    # generates a name based off name of edit rotation and what position in axis vector dstr data was
    def getEditedName(self, dstr, axis, nmod):
        return f'{dstr}*' if nmod=='*' else f'{axis}{nmod}' #{dstr[:2]} somewhere could add first 2 characters of dstr, usually pretty descriptive

    # given current axis vector selections
    # make sure that all the correct data is calculated with matrix R
    def generateData(self, R, nmod=None):
        r = Mth.matToString(R)
        i = Mth.identity()
        
        # for each full vector dropdown row 
        for di, dd in enumerate(self.axisDropdowns):
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
                X = self.window.DATADICT[xstr][i][0]
                Y = self.window.DATADICT[ystr][i][0]
                Z = self.window.DATADICT[zstr][i][0]

                A = np.column_stack((X,Y,Z))
                M = np.matmul(A,R)

                self.window.DATADICT[xstr][r] = [M[:,0], xstr, f'X{di+1}']
                self.window.DATADICT[ystr][r] = [M[:,1], ystr, f'Y{di+1}']
                self.window.DATADICT[zstr][r] = [M[:,2], zstr, f'Z{di+1}']


class ManRot(QtWidgets.QFrame, ManRotUI):
    def __init__(self, edit, window, parent=None):
        super(ManRot, self).__init__(parent)#, QtCore.Qt.WindowStaysOnTopHint)
        self.edit = edit
        self.window = window

        self.ui = ManRotUI()
        self.ui.setupUI(self, window)

        self.FLAG = 1.e-10
        self.D2R = pi / 180.0 # degree to rad conversion constant

        for i,gb in enumerate(self.ui.genButtons):
            gb.clicked.connect(functools.partial(self.axisRotGen, Mth.AXES[i]))

        self.ui.loadIdentity.clicked.connect(functools.partial(Mth.setMatrix, self.ui.R, Mth.IDENTITY))
        self.ui.loadZeros.clicked.connect(functools.partial(Mth.setMatrix, self.ui.R, Mth.empty()))
        self.ui.loadCurrentEditMatrix.clicked.connect(self.loadCurrentEditMatrix)
        self.ui.applyButton.clicked.connect(self.apply)

        self.lastOpName = 'Custom'

        Mth.setMatrix(self.ui.R, Mth.IDENTITY)
        
    def apply(self):
        # figure out if custom on axisrot
        self.edit.apply(Mth.getMatrix(self.ui.R), '', self.lastOpName)
        Edit.moveToFront(self.edit)

    def axisRotGen(self, axis):
        angle = self.ui.axisAngle.value()
        R = self.genAxisRotationMatrix(axis, angle)
        Mth.setMatrix(self.ui.R, R)
        self.lastOpName = f'{axis}{angle}rot'

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
        return Mth.copy(Mth.IDENTITY)

    def loadCurrentEditMatrix(self):
        mat = Mth.getMatrix(self.edit.ui.M)
        Mth.setMatrix(self.ui.R, mat)


class MinVar(QtWidgets.QFrame, MinVarUI):
    def __init__(self, edit, window, parent=None):
        super(MinVar, self).__init__(parent)#, QtCore.Qt.WindowStaysOnTopHint)
        self.edit = edit
        self.window = window

        self.ui = MinVarUI()
        self.ui.setupUI(self, window)

        self.window.startGeneralSelect('MINVAR', '#0000FF', self.ui.timeEdit)

        self.ui.applyButton.clicked.connect(self.calcMinVar)

        self.shouldResizeWindow = False

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

        iO,iE = self.window.getTicksFromLines()

        xyz = []
        avg = []
        for v in self.ui.vector:
            data = self.window.getData(v.currentText())[iO:iE]
            xyz.append(data)
            avg.append(self.average(data))

        items = len(xyz[0])

        covar = Mth.empty()
        CoVar = Mth.empty()
        eigen = Mth.empty()

        for i in Mth.i:
            for j in Mth.i[i:]:
                for k in range(items):
                    covar[i][j] = covar[i][j] + xyz[i][k] * xyz[j][k]
        for i in Mth.i:
            for j in Mth.i[i:]:
                CoVar[i][j] = (covar[i][j] / items) - avg[i] * avg[j]
        # fill out lower triangle
        CoVar[1][0] = CoVar[0][1]
        CoVar[2][0] = CoVar[0][2]
        CoVar[2][1] = CoVar[1][2]

        A = np.array(CoVar)
        w, v = np.linalg.eigh(A, UPLO="U")

        for i in Mth.i:
            for j in Mth.i:
                eigen[2 - i][j] = v[j][i]
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
        

        ev = [w[2], w[1], w[0]]
        prec = 6
        eigenText = f'EigenVals: {ev[0]:.{prec}f}, {ev[1]:.{prec}f}, {ev[2]:.{prec}f}'
        self.ui.eigenValsLabel.setText(eigenText)
        self.edit.apply(eigen, eigenText, 'minvar')
        Edit.moveToFront(self.edit)
        #print('min var calculation completed')
        #print(ev)
