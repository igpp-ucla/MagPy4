
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import pyqtgraph as pg
from ..MagPy4UI import ScientificSpinBox
from ..dispwidgets.layouttools import TimeEdit
from ..plotbase import StackedLabel

import numpy as np
from math import sin, cos, acos, fabs, pi
from scipy import signal

import functools
import time

from .editui import EditUI, CustomRotUI, MinVarUI
from .filterdialog import filterdialog
from .calctool import simpleCalc

from ..alg.mth import Mth
from ..MagPy4UI import PyQtUtils

from datetime import datetime
from ..geopack.geopack import geopack
from fflib import ff_time

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

        vecArrays = np.array(found).T
        if Mth.flattenLst(found, 1) == []:
            vecArrays = []
            self.ui.vecLt.setDefVecs([])
            self.ui.vecLt.buildDropdowns([self.window.DATASTRINGS[0:3]])
        else:
            self.ui.vecLt.setDefVecs(vecArrays)
            self.ui.vecLt.buildDropdowns(vecArrays)

        self.axisDropdowns = self.ui.vecLt.dropdowns

        self.history = [] # lists of edit matrix, and string for extra data (ie eigenvalues)
        if self.window.editHistory:
            for h,name in self.window.editHistory:
                if h:
                    self.addHistory(h[0], h[1], name)
                else:
                    self.ui.history.setCurrentRow(name)
        else:
            self.addHistory(Mth.IDENTITY, 'Original Data', 'Identity')

        self.ui.history.currentRowChanged.connect(self.onHistoryChanged)
        self.ui.history.itemChanged.connect(self.onHistoryChanged)

        self.onHistoryChanged()

        self.minVar = None
        self.ui.minVarButton.clicked.connect(self.openMinVar)
        self.customRot = None
        self.ui.customRotButton.clicked.connect(self.openCustomRot)
        self.filter = None
        self.ui.filterButton.clicked.connect(self.openFilter)
        self.simpCalc = None
        self.ui.calcBtn.clicked.connect(self.openSimpleCalc)

        self.dataFlagTool = None
        self.ui.dataFlagBtn.clicked.connect(self.openDataFlagTool)

        self.smoothTool = None
        if window.insightMode:
            self.ui.smoothBtn.clicked.connect(window.startSmoothing)

        self.gsm_gse_tool = None
        self.ui.gseGsmBtn.clicked.connect(self.startGSMGSE)

    def startGSMGSE(self):
        ''' Open GSM/GSE conversion tool '''
        self.closeGSMGSE()
        self.gsm_gse_tool = GSM_GSE_Tool(self.window, self)
        self.gsm_gse_tool.show()
    
    def closeGSMGSE(self):
        ''' Close GSM/GSE conversion tool '''
        if self.gsm_gse_tool:
            self.gsm_gse_tool.close()
            self.gsm_gse_tool = None

    def closeEvent(self, event):
        # Save edit history
        self.window.editHistory = self.getEditHistory()

        self.closeCustomRot()
        self.closeMinVar()
        self.closeSimpleCalc()
        self.closeDataFlagTool()
        self.closeGSMGSE()

    def getEditHistory(self):
        hist = []
        for i in range(len(self.history)):
            hist.append((self.history[i],self.ui.history.item(i).text()))
        hist.append(([], self.ui.history.currentRow())) # save row that was selected as last element
        return hist

    def closeSubWindows(self):
        self.closeCustomRot()
        self.closeMinVar()
        self.closeFilter()
        self.closeSimpleCalc()

    def openDataFlagTool(self):
        self.closeDataFlagTool()

        # Minimize this window but keep running
        self.showMinimized()

        # Open data flag tool and start general select
        self.dataFlagTool = DataFlagTool(self.window, self)
        self.window.initGeneralSelect('Data Removal', '#2c1e45', 
            self.dataFlagTool.timeEdit, 'Multi', maxSteps=-1)
        self.dataFlagTool.show()

    def closeDataFlagTool(self):
        if self.dataFlagTool:
            self.dataFlagTool.close()
            self.dataFlagTool = None

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
        self.filter = filterdialog(self, self.window)
        self.window.initGeneralSelect('Filter', '#32a852', self.filter.ui.timeEdit,
            'Single', closeFunc=self.closeFilter)
        self.filter.finished.connect(self.window.endGeneralSelect)
        self.filter.show()

    def closeFilter(self):
        if self.filter:
            self.filter.close()
            self.filter = None

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

    # removes selected history
    def removeHistory(self):
        curRow = self.ui.history.currentRow()
        if curRow == 0:
            print('cannot remove original data')
            return

        for dstr,datas in self.window.DATADICT.items():
            del datas[curRow]
            # Remove any edited times corresponding to this edit
            if dstr in self.window.EDITEDTIMES:
                timeDict = self.window.EDITEDTIMES[dstr]
                if curRow in timeDict:
                    del timeDict[curRow]

        self.ui.history.setCurrentRow(curRow - 1) # change before take item otherwise onHistory gets called with wrong row
        self.ui.history.takeItem(curRow)
        del self.history[curRow]

    def onHistoryChanged(self):
        row = self.ui.history.currentRow()
        self.curSelection = self.history[row]
        old_edit = self.window.currentEdit
        self.window.currentEdit = row
        self.ui.M.setMatrix(self.curSelection[0])
        self.ui.extraLabel.setText(self.curSelection[1])

        # Rebuild edit name list
        old_names = self.window.editNames[:]
        self.window.editNames = [self.ui.history.item(i).text() for i in range(self.ui.history.count())]

        # Update plot edits
        if row != old_edit or old_names != self.window.editNames:
            self.window.update_current_edit(old_edit, row) 

    # takes a matrix, notes for the history, and a name for the history entry
    # axisVecs defines vectors to modify instead of dropdown selections
    def apply(self, mat, notes, name, multType='R'):
        # shows total matrix from beginning
        if multType == 'R':
            R = Mth.mult(self.curSelection[0], mat)
        else:
            R = Mth.mult(mat, self.curSelection[0])
        self.generateData(mat, name, multType)
        self.addHistory(R, notes, f'{name}')

    # given current axis vector selections
    # make sure that all the correct data is calculated with matrix R
    def generateData(self, R, name, multType='R'):
        vectorList = self.ui.vecLt.getDropdownVecs()
        # for each full vector dropdown row 
        for di, dstrs in enumerate(vectorList):
            xstr = dstrs[0]
            ystr = dstrs[1]
            zstr = dstrs[2]

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
        if self.window.insightMode:
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
        theta = 57.9 * self.D2R
        mat = np.array([[cos(theta), sin(theta), 0],
                        [-sin(theta), cos(theta), 0],
                        [0, 0, 1]])
        self.ui.R.setMatrix(mat)
        self.lastOpName = 'SC'

    def loadSpaceToLocMat(self):
        mat = np.array([[0.99886589, -0.00622313, 0.04720395],
                        [0.00862136, 0.99867298, -0.05077360],
                        [-0.04682533, 0.05112297, 0.99759402]])
        self.ui.R.setMatrix(mat)
        self.lastOpName = 'LL'

class MinVar(QtWidgets.QFrame, MinVarUI):
    def __init__(self, edit, window, parent=None):
        super(MinVar, self).__init__(parent)#, QtCore.Qt.WindowStaysOnTopHint)
        self.edit = edit
        self.window = window

        self.ui = MinVarUI()
        self.ui.setupUI(self, window)

        self.window.initGeneralSelect('Min Var', '#ffbf51', self.ui.timeEdit,
            'Single', startFunc=None, closeFunc=self.close)

        self.ui.applyButton.clicked.connect(self.calcMinVar)

        self.shouldResizeWindow = False

    def closeEvent(self, event):
        self.window.endGeneralSelect()
        self.close()

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
        PyQtUtils.moveToFront(self.edit)
        self.edit.closeMinVar()

class DataFlagTool(QtWidgets.QFrame):
    def __init__(self, window, editFrame):
        self.window = window
        self.editFrame = editFrame
        QtWidgets.QFrame.__init__(self)
        self.setupUI(window)
        self.applyBtn.clicked.connect(self.flagData)
        self.toggleWindowOnTop(True)

    def toggleWindowOnTop(self, val):
        self.setParent(self.window if val else None)
        dialogFlag = QtCore.Qt.Dialog
        if self.window.OS == 'posix':
            dialogFlag = QtCore.Qt.Tool
        flags = self.windowFlags()
        flags = flags | dialogFlag if val else flags & ~dialogFlag
        self.setWindowFlags(flags)
        self.show()

    def setupUI(self, window):
        layout = QtWidgets.QGridLayout(self)
        self.setWindowTitle('Data Removal')

        # Set up time edit
        timeLt = QtWidgets.QHBoxLayout()
        self.timeEdit = TimeEdit()
        timeLt.addWidget(self.timeEdit.start)
        timeLt.addWidget(self.timeEdit.end)
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())

        # Set up operation radio buttons
        optionsFrame = QtWidgets.QGroupBox('Operation:')
        optionsLt = QtWidgets.QHBoxLayout(optionsFrame)
        optionsLt.setAlignment(QtCore.Qt.AlignTop)

        self.flagCheck = QtWidgets.QRadioButton('Fill with flag:')
        self.flagBox = ScientificSpinBox()
        self.flagBox.setMinimum(-1e32)
        self.flagBox.setMaximum(1e32)
        self.flagBox.setValue(1e32)

        self.deleteCheck = QtWidgets.QRadioButton('Remove data')
        self.deleteCheck.setChecked(True)

        # Add buttons to layout
        for elem in [self.deleteCheck, self.flagCheck, self.flagBox]:
            optionsLt.addWidget(elem)
        optionsLt.addStretch()
        optionsLt.setSpacing(10)

        # Apply button and general layout structure
        self.applyBtn = QtWidgets.QPushButton('Apply')
        layout.addLayout(timeLt, 1, 0, 1, 1)
        layout.addWidget(optionsFrame, 0, 0, 1, 2)
        layout.addWidget(self.applyBtn, 1, 1, 1, 1)

    def flagData(self):
        if len(self.window.FIDs) > 1:
            return

        # Get flag, use None if removing data completely
        if self.flagCheck.isChecked():
            flag = self.flagBox.value()
        else:
            flag = None

        # Call window function to flag or remove data
        regionTicks = self.window.flagData(flag)

        # Convert region ticks to timestamps
        mapTick = lambda t : self.window.getTimestampFromTick(t)
        timestamps = [(mapTick(t0), mapTick(t1)) for t0, t1 in regionTicks]

        # Add to edit history
        name = 'Removal'
        mat = np.eye(3)
        notes = 'Removed data at:\n'
        notes += '\n'.join([f'{ts[0]}, {ts[1]}' for ts in timestamps])
        self.editFrame.addHistory(mat, notes, name)

        # Close selected regions
        self.window.currSelect.closeAllRegions()
    
    def closeEvent(self, ev):
        # If edit window is still minimized, close it
        if self.editFrame.isMinimized():
            self.editFrame.close()

        # End general select and close
        self.window.endGeneralSelect()
        self.close()

class GSM_GSE_Coord():
    TO_GSM = -1
    TO_GSE = 1
    def get_universal_time(epoch, ticks):
        ''' Convert seconds since epoch time to
            universal time
        '''
        # Create lambdas to convert ticks to datetimes and
        # then to the univeral time
        udt = datetime(1970, 1, 1)
        to_dt = lambda x : ff_time.tick_to_date(x, epoch)
        to_diff = lambda t : (to_dt(t) - udt).total_seconds()

        # Compute map
        uttimes = list(map(to_diff, ticks))

        return uttimes

    def map_coords(x, y, z, t, epoch, direction):
        ''' Map x, y, z coordinates with t seconds since epoch
            from gsm to gse or gse to gsm depending on direction
        '''
        # Map time ticks to universal time
        ut = GSM_GSE_Coord.get_universal_time(epoch, t)

        # Convert coordinates and yield in tuple format
        vals = []
        for xt, yt, zt, tt in zip(x, y, z, ut):
            geopack.recalc(tt)
            res = geopack.gsmgse(xt, yt, zt, direction)
            vals.append(res)
        
        return vals

    def map_to_recs(coords):
        ''' Map list of tuples to a structured numpy records array '''
        dtype = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
        return np.array(coords, dtype=dtype)

    def gsm_to_gse(x, y, z, t, epoch):
        ''' Map x, y, z coordinates with t seconds since epoch
            from GSM to GSE
        '''
        d = GSM_GSE_Coord.TO_GSE
        coords = list(GSM_GSE_Coord.map_coords(x, y, z, t, epoch, d))
        records = GSM_GSE_Coord.map_to_recs(coords)
        return records

    def gse_to_gsm(x, y, z, t, epoch):
        ''' Map x, y, z coordinates with t seconds since epoch
            from GSE to GSM
        '''
        d = GSM_GSE_Coord.TO_GSM
        coords = list(GSM_GSE_Coord.map_coords(x, y, z, t, epoch, d))
        records = GSM_GSE_Coord.map_to_recs(coords)
        return records

class GSM_GSE_Tool(QtWidgets.QFrame):
    def __init__(self, window, editWindow):
        self.window = window
        self.editWindow = editWindow

        super().__init__()
        self.setupLayout()
        self.modes = {
            'GSM to GSE' : GSM_GSE_Coord.TO_GSE,
            'GSE to GSM' : GSM_GSE_Coord.TO_GSM,
        }
        self.applyBtn.clicked.connect(self.apply)

    def setupLayout(self):
        self.setWindowTitle('GSM/GSE Conversion')
        layout = QtWidgets.QVBoxLayout(self)

        # Vector layout
        self.vecBoxes = [QtWidgets.QComboBox() for i in range(3)]

        ## Get vector groups from main window and add to boxes
        for grp in self.window.VECGRPS:
            vec = self.window.VECGRPS[grp]
            vec = vec[:3]
            if len(vec) < 3:
                continue

            for dstr, box in zip(vec, self.vecBoxes):
                box.addItem(dstr)

        vecLt = QtWidgets.QHBoxLayout()
        for vecBox in self.vecBoxes:
            vecLt.addWidget(vecBox)

        # Direction option
        items = ['GSM to GSE', 'GSE to GSM']
        self.optsBox = QtWidgets.QComboBox()
        self.optsBox.addItems(items)

        optsLt = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel('Mode: ')
        label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        optsLt.addWidget(label)
        optsLt.addWidget(self.optsBox)

        # Apply button
        self.applyBtn = QtWidgets.QPushButton('Apply')
        applyLt = QtWidgets.QHBoxLayout()
        applyLt.addStretch()
        applyLt.addWidget(self.applyBtn)
        
        layout.addLayout(vecLt)
        layout.addLayout(optsLt)
        layout.addLayout(applyLt)

    def gsm_gse_conv(self, vecs, to_gse=True):
        ''' Converts vectors to GSM or GSE based on to_gsm flag 
            and adds an entry to the edit list
        '''
        # Get conversion function
        func = GSM_GSE_Coord.gse_to_gsm
        if to_gse:
            func = GSM_GSE_Coord.gsm_to_gse

        # Iterate over vectors
        for vec in vecs:
            # Get data
            en = self.window.currentEdit
            data = [self.window.getData(dstr, en) for dstr in vec]
            x, y, z = data

            # Get times
            times = self.window.getTimes(vec[0], en)[0]
            epoch = self.window.epoch

            # Convert coordinates
            new_data = func(x, y, z, times, epoch)
            keys = new_data.dtype.names

            # Add to history
            for dstr, key in zip(vec, keys):
                self.window.DATADICT[dstr].append(new_data[key])

        # Add edit entry
        label = '2gse' if to_gse else '2gsm'
        notes = 'GSM to GSE' if to_gse else 'GSE to GSM'
        self.editWindow.addHistory(np.eye(3), notes, label)

    def apply(self):
        ''' Apply changes based on UI settings '''
        # Get vector to apply change to
        vec = [box.currentText() for box in self.vecBoxes]

        # Get conversion direction
        mode = self.optsBox.currentText()
        mode = self.modes[mode]
        to_gse = (mode == GSM_GSE_Coord.TO_GSE)

        # Convert data
        self.gsm_gse_conv([vec], to_gse)