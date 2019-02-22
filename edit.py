
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from MagPy4UI import StackedLabel

#import pyqtgraph as pg
import numpy as np
from FF_Time import FFTIME
from math import sin, cos, acos, fabs, pi

import functools
import time

from editUI import EditUI, CustomRotUI, MinVarUI
from FilterDialog import FilterDialog
from simpleCalculations import simpleCalc

from mth import Mth
from MagPy4UI import PyQtUtils

class Edit(QtWidgets.QFrame, EditUI):

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

        self.history = [] # lists of edit matrix, and string for extra data (ie eigenvalues)
        if self.window.editHistory:
            for h,name in self.window.editHistory:
                if h:
                    self.addHistory(h[0], h[1], name)
                else:
                    self.ui.history.setCurrentRow(name)
        else:
            self.addHistory(Mth.IDENTITY, 'original data', 'Identity')

        self.ui.history.currentRowChanged.connect(self.onHistoryChanged)

        self.onHistoryChanged(self.ui.history.currentRow())

        self.minVar = None
        self.ui.minVarButton.clicked.connect(self.openMinVar)
        self.customRot = None
        self.ui.customRotButton.clicked.connect(self.openCustomRot)
        self.filter = None
        self.ui.filterButton.clicked.connect(self.openFilter)
        self.simpCalc = None
        self.ui.calcBtn.clicked.connect(self.openSimpleCalc)

    def closeEvent(self, event):
        # save edit history
        hist = []
        for i in range(len(self.history)):
            hist.append((self.history[i],self.ui.history.item(i).text()))
            #print(f'{hist[i][0]} {hist[i][1]}' )
        hist.append(([], self.ui.history.currentRow())) # save row that was selected as last element

        self.window.editHistory = hist

        self.closeCustomRot()
        self.closeMinVar()
        self.closeSimpleCalc()

    def closeSubWindows(self):
        self.closeCustomRot()
        self.closeMinVar()
        self.closeFilter()
        self.closeSimpleCalc()

    def openCustomRot(self):
        self.closeSubWindows()
        self.customRot = CustomRot(self, self.window)
        self.customRot.show()

    def closeCustomRot(self):
        if self.customRot:
            self.customRot.close()
            self.customRot = None

    def openMinVar(self):
        self.closeSubWindows()
        self.minVar = MinVar(self, self.window)
        self.minVar.show()

    def closeMinVar(self):
        if self.minVar:
            self.minVar.close()
            self.minVar = None

    def openSimpleCalc(self):
        self.closeSubWindows()
        self.simpCalc = simpleCalc(self, self.window)
        self.simpCalc.show()

    def closeSimpleCalc(self):
        if self.simpCalc:
            self.simpCalc.close()
            self.simpCalc = None

    def openFilter(self):
        self.closeSubWindows()
        self.filter = FilterDialog(self, self.window)
        self.filter.show()

    def closeFilter(self):
        if self.filter:
            self.filter.close()
            self.filter = None

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

    # adds an entry to history matrix list and a list item at end of history ui
    def addHistory(self, mat, notes, name):
        self.history.append([Mth.copy(mat), notes])

        # pad the rest of datadict to have same length
        length = len(self.history)
        for k,v in self.window.DATADICT.items():
            while len(v) < length:
                v.append([])

        # get names of items
        uihist = self.ui.history
        taken = set()
        for i in range(uihist.count()):
            taken.add(uihist.item(i).text())

        # ensure no duplicate names are added to history
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
        self.updateYLabels(uihist.count() - 1)

    # removes selected history
    def removeHistory(self):
        curRow = self.ui.history.currentRow()
        if curRow == 0:
            print('cannot remove original data')
            return

        for dstr,datas in self.window.DATADICT.items():
            del datas[curRow]

        self.ui.history.setCurrentRow(curRow - 1) # change before take item otherwise onHistory gets called with wrong row
        self.ui.history.takeItem(curRow)
        del self.history[curRow]

    def updateYLabels(self, currentEdit):
        """Updates Y axis label strings for each plot"""
        plotNum = 0
        for dstrs,pens in zip(self.window.lastPlotStrings,self.window.plotTracePens):
            previousLabel = self.window.pltGrd.getPlotLabel(plotNum)
            labels = []
            colors = previousLabel.colors[:]
            units = previousLabel.units
            for (dstr,editNum),pen in zip(dstrs,pens):
                l = self.window.getLabel(dstr, currentEdit)
                if l in self.window.ABBRV_DSTR_DICT:
                    l = self.window.ABBRV_DSTR_DICT[l]
                labels.append(l)
            newLabel = StackedLabel(labels, colors, units=units)
            
            self.window.pltGrd.setPlotLabel(newLabel, plotNum)
            plotNum += 1

    def onHistoryChanged(self, row):
        self.curSelection = self.history[row]
        self.window.currentEdit = row
        self.ui.M.setMatrix(self.curSelection[0])
        self.ui.extraLabel.setText(self.curSelection[1])

        # rebuild edit name list
        self.window.editNames = [self.ui.history.item(i).text() for i in range(self.ui.history.count())]

        self.window.replotData(row) # provide row (which is the edit number) so plotter can try to swap things to that automatically
        self.updateYLabels(row)

    # takes a matrix, notes for the history, and a name for the history entry
    # axisVecs defines vectors to modify instead of dropdown selections
    def apply(self, mat, notes, name, multType='R', axisVecs=None):
        # shows total matrix from beginning
        if multType == 'R':
            R = Mth.mult(self.curSelection[0], mat)
        else:
            R = Mth.mult(mat, self.curSelection[0])
        self.generateData(mat, name, multType, axisVecs)
        self.addHistory(R, notes, f'{name}')

    # given current axis vector selections
    # make sure that all the correct data is calculated with matrix R
    def generateData(self, R, name, multType='R', axisVecs=None):
        vectorList = self.axisDropdowns if axisVecs is None else axisVecs
        # for each full vector dropdown row 
        for di, dd in enumerate(vectorList):
            xstr = dd[0].currentText()
            ystr = dd[1].currentText()
            zstr = dd[2].currentText()

            if not xstr or not ystr or not zstr: # skip rows with empty selections
                continue

            # multiply currently selected data by new matrix
            X = self.window.getData(xstr)
            Y = self.window.getData(ystr)
            Z = self.window.getData(zstr)

            # If a right matrix transformation, treat data vecs as columns
            if (multType == 'R'):
                A = np.column_stack((X,Y,Z))
                M = np.matmul(A,R)
            else:
            # If a left matrix transformation, treat data vecs as rows and transpose
            # result for compatability with next few lines of code
                A = np.array([X,Y,Z])
                Mtran = np.matmul(R,A)
                M = np.transpose(Mtran)

            self.window.DATADICT[xstr].append(M[:,0])
            self.window.DATADICT[ystr].append(M[:,1])
            self.window.DATADICT[zstr].append(M[:,2])

class CustomRot(QtWidgets.QFrame, CustomRotUI):
    def __init__(self, edit, window, parent=None):
        super(CustomRot, self).__init__(parent)#, QtCore.Qt.WindowStaysOnTopHint)
        self.edit = edit
        self.window = window

        self.ui = CustomRotUI()
        self.ui.setupUI(self, window)

        self.FLAG = 1.e-10
        self.D2R = pi / 180.0 # degree to rad conversion constant

        self.ui.loadIdentity.clicked.connect(functools.partial(self.ui.R.setMatrix, Mth.IDENTITY))
        self.ui.loadZeros.clicked.connect(functools.partial(self.ui.R.setMatrix, Mth.empty()))
        self.ui.loadCurrentEditMatrix.clicked.connect(self.loadCurrentEditMatrix)

        self.ui.axisAngle.valueChanged.connect(self.axisAngleChanged)

        self.axisChecked = None

        for i,gb in enumerate(self.ui.genButtons):
            gb.clicked.connect(functools.partial(self.axisRotGen, Mth.AXES[i]))
            if gb.isChecked():
                self.axisChecked = gb.text()

        self.ui.applyButton.clicked.connect(self.apply)
        self.ui.spaceToLocBtn.clicked.connect(self.loadSpaceToLocMat)
        self.ui.instrToSpaceBtn.clicked.connect(self.loadInstrToSpaceMat)

        self.lastOpName = 'Custom'

        self.ui.R.setMatrix(Mth.IDENTITY)
        
    def apply(self):
        # figure out if custom on axisrot
        self.edit.apply(self.ui.R.getMatrix(), '', self.lastOpName, 'L')
        self.edit.closeCustomRot()
        PyQtUtils.moveToFront(self.edit)

    def axisRotGen(self, axis):
        self.axisChecked = axis
        angle = self.ui.axisAngle.value()
        R = self.genAxisRotationMatrix(axis, angle)
        self.ui.R.setMatrix(R)
        self.lastOpName = f'{axis}{angle}rot'

    def axisAngleChanged(self):
        self.axisRotGen(self.axisChecked)

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
        self.ui.R.setMatrix(self.edit.ui.M.getMatrix())

    def loadInstrToSpaceMat(self):
        theta = 57.9
        mat = np.array([[cos(theta), -sin(theta), 0],
                        [sin(theta), cos(theta), 0],
                        [0, 0, 1]])
        self.ui.R.setMatrix(mat)

    def loadSpaceToLocMat(self):
        mat = np.array([[0.99886589, -0.00622313, 0.04720395],
                        [0.00862136, 0.99867298, -0.05077360],
                        [-0.04682533, 0.05112297, 0.99759402]])
        self.ui.R.setMatrix(mat)

class MinVar(QtWidgets.QFrame, MinVarUI):
    def __init__(self, edit, window, parent=None):
        super(MinVar, self).__init__(parent)#, QtCore.Qt.WindowStaysOnTopHint)
        self.edit = edit
        self.window = window

        self.ui = MinVarUI()
        self.ui.setupUI(self, window)

        self.window.initGeneralSelect('Min Var', '#ffbf51', self.ui.timeEdit)

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
        # otherwise minvar calcs may get messed up if smoothed data affects the average

        xyz = []
        avg = []
        vstrs = []

        for v in self.ui.vector:
            vstr = v.currentText()
            vstrs.append(vstr)
            iO,iE = self.window.calcDataIndicesFromLines(vstr, self.window.currentEdit)
            data = self.window.getData(vstr)[iO:iE]
            xyz.append(data)
            avg.append(self.average(data))

        items = len(xyz[0])

        covar = Mth.empty()
        CoVar = Mth.empty()
        eigen = Mth.empty()

        for i in range(3):
            for j in range(3)[i:]:
                for k in range(items):
                    covar[i][j] = covar[i][j] + xyz[i][k] * xyz[j][k]
        for i in range(3):
            for j in range(3)[i:]:
                CoVar[i][j] = (covar[i][j] / items) - avg[i] * avg[j]
        # fill out lower triangle
        CoVar[1][0] = CoVar[0][1]
        CoVar[2][0] = CoVar[0][2]
        CoVar[2][1] = CoVar[1][2]

        A = np.array(CoVar)
        w, v = np.linalg.eigh(A, UPLO="U")
        #print(f'{w}\n {v}')

        for i in range(3):
            for j in range(3):
                eigen[2 - i][j] = v[j][i]
        # force to be right handed
        e20 = eigen[0][1] * eigen[1][2] - eigen[0][2] * eigen[1][1]
        if e20 * eigen[2][0] < 0:
            eigen[2] = np.negative(eigen[2])
        if eigen[0][2] < 0.:
            eigen[0] = np.negative(eigen[0])
            eigen[1] = np.negative(eigen[1])
        if eigen[2][0] < 0.:
            eigen[1] = np.negative(eigen[1])
            eigen[2] = np.negative(eigen[2])

        ev = [w[2], w[1], w[0]]
        prec = 6
        eigenText = f'EigenVals: {ev[0]:.{prec}f}, {ev[1]:.{prec}f}, {ev[2]:.{prec}f}'
        self.ui.eigenValsLabel.setText(eigenText)
        ts = self.ui.timeEdit.toString()
        labelText = f'{", ".join(vstrs)}\n{eigenText}\n{ts[0]}->{ts[1]}'
        self.edit.apply(eigen, labelText, 'MinVar', 'L')
        #self.edit.closeMinVar()
        PyQtUtils.moveToFront(self.edit)
