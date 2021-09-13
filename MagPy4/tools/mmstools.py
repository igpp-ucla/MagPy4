from ..plotbase.grid import PlotGridObject
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from ..MagPy4UI import MatrixWidget, VectorWidget
from ..qtinterface.layouttools import TimeEdit

from ..plotbase import MagPyPlotItem, StackedLabel
from ..qtinterface.layouttools import BaseLayout

from ..qtthread import TaskRunner

import scipy
from scipy import constants
from scipy.interpolate import CubicSpline

import pyqtgraph as pg
import numpy as np
import bisect

def key_search(key, d):
    for comp_key in d:
        if key in comp_key:
            return comp_key
    return None

def get_mms_grps(window, coords_mode=None):
    ''' Get groups of variable names by vector '''
    # Determine coordinate system being used
    coords = 'gsm'
    for lst in window.lastPlotStrings:
        for dstr, en in lst:
            if 'gse' in dstr.lower():
                coords = 'gse'
    if coords_mode is not None:
        coords = coords

    # Get position and field variables
    btot_dstrs = []
    full_grps = {}
    lowered_grps = {key.lower():key for key in window.VECGRPS}
    for start_key, name in [('R', 'Pos'), ('B', 'Field')]:
        grps = {}
        full_grps[name] = grps
        for sc_id in [1,2,3,4]:
            key = f'mms{sc_id}_fgm_{start_key}_{coords}'.lower()
            base_key = key_search(key, lowered_grps)
            if base_key:
                grp = window.VECGRPS[base_key]
                grps[sc_id] = grp
                if len(grp) == 4:
                    btot_dstrs.append(grp[-1])

    return full_grps, btot_dstrs 

class MMSTools():
    def __init__(self, window):
        self.earthRadius = 6371
        self.mmsState = True
        self.diffMode = False
        self.window = window
        # Dicts mapping field/pos kws to datastrings
        self.grps = {} # Internally maps by field axis (bx1, bx2, ...)
        self.scGrps = {} # Internally maps by spacecraft number (bx1, by1, ...)
        self.btotDstrs = []
        self.initGroups()

        if self.mmsState == False:
            self.raiseErrorMsg()
            return

        self.pool = QtCore.QThreadPool()
        worker = TaskRunner(self.initArrays)
        self.pool.start(worker)

    def waitForData(self):
        while (self.pool.activeThreadCount()) > 0:
            pass

    def inValidState(self):
        return self.mmsState

    def raiseErrorMsg(self):
        self.window.ui.statusBar.showMessage('Error: Missing MMS Data')

    def getDstrsBySpcrft(self, scNum, grp='Field', magn=False):
        data = self._clip_magn(self.scGrps[grp][scNum], magn)
        return data

    def getDstrsByVec(self, vec, grp='Field', magn=False):
        data = self._clip_magn(self.grps[grp][vec], magn)
        return data

    def getVec(self, scNum, index, grp='Field', magn=False):
        data = self._clip_magn(self.vecArrays[grp][scNum][:,index], magn)
        return data

    def _clip_magn(self, data, magn=False):
        if len(data) == 4 and magn == False:
            return data[:-1]
        return data

    def getPosData(self, scNum, startIndex=None, endIndex=None, radius=False):
        if startIndex is None:
            startIndex = 0
        if endIndex is None:
            endIndex = len(self.vecArrays['Pos'][scNum][0])
        data = self.vecArrays['Pos'][scNum][:,startIndex:endIndex]
        if radius is False and len(data) == 4:
            return data[:-1]
        return data

    def getMagData(self, scNum, startIndex=None, endIndex=None, btot=False):
        if startIndex is None:
            startIndex = 0
        if endIndex is None:
            endIndex = len(self.vecArrays['Field'][scNum][0])
        data = self.vecArrays['Field'][scNum][:,startIndex:endIndex]
        if btot is False and len(data) == 4:
            return data[:-1]
        return data

    def initArrays(self):
        '''
        Creates dictionary of arrays s.t. a column within the array
        corresponds to a field/position vector at a given data index
        i.e. vecArrays['Field'][4] = [[bx4 data] [by4 data] [bz4 data]]
        '''
        # Check if data previously interpolated
        if self.checkForPrev():
            self.vecArrays = {k:v for k, v in self.window.mmsInterp.items()}
            return

        self.vecArrays = {}

        refTimes = None
        for grp in ['Field', 'Pos']:
            self.vecArrays[grp] = {}
            for scNum in [1,2,3,4]:
                dstrs = self.getDstrsBySpcrft(scNum, grp)
                vecArr = []
                axisIndex = 0
                for dstr in dstrs:
                    dta = self.window.getData(dstr)
                    times = self.window.getTimes(dstr, 0)[0]

                    # Interpolate data along MMS1 times using Cubic Splines
                    if refTimes is None:
                        refTimes = times
                    else:
                        cs = CubicSpline(times, dta)
                        dta = cs(refTimes)
                    vecArr.append(dta)
                    axisIndex += 1
                self.vecArrays[grp][scNum] = np.array(vecArr)

        # Update in main window
        self.window.mmsInterp.update(self.vecArrays)

    def checkForPrev(self):
        # Check if previous interpolations are empty
        mmsInterp = self.window.mmsInterp
        if len(self.window.mmsInterp) == 0:
            return False

        # Get keyword for MMS1 FGM data
        mms1Kw = self.scGrps['Field'][1][0]
        data = self.window.getData(mms1Kw, 0)

        # Check if interpolated data and loaded MMS1 FGM data have same length
        if 'Field' not in mmsInterp or len(mmsInterp['Field']) == 0:
            return False

        interp = mmsInterp['Field'][1][0]
        if len(interp) != len(data):
            return False

        return True

    def initGroups(self):
        self.scGrps = self.getMMSGrps()
        self.grps = self.getAxisGrps(self.scGrps)

    def getMMSGrps(self):
        full_grps, self.btotDstrs = get_mms_grps(self.window)
        return full_grps

    def getAxisGrps(self, scGrps):
        ''' Group variables by axes '''
        full_axes_grps = {}
        # Iterate over data type groups
        for grp in scGrps:
            grps = scGrps[grp]
            axes = ['X', 'Y', 'Z']
            axes_grp = {axis:[] for axis in axes}
            full_axes_grps[grp] = axes_grp
            # Iterate over spacecraft groups (BX1, BY1, BZ1), (BX2, BY2,..),
            for sc_id in grps:
                sc_grp = grps[sc_id]

                # Create groups from each axis in the spacecraft group
                # so we have groups like (BX1, BX2, BX3, BX4)
                i = 0
                while i < len(sc_grp) and i < len(axes):
                    axis = axes[i]
                    axes_grp[axis].append(sc_grp[i])
                    i += 1

        return full_axes_grps

    def getFPITempPerpKws(self):
        # Determine what keywords should be found for ion/electron data
        kws = {'I':[], 'E':[]}
        for sc_id in [1,2,3,4]:
            for data_type in kws:
                kw = f'MMS{sc_id}_FPI{sc_id}/D{data_type}S_tempPerp'
                kws[data_type].append(kw)

        # Determine which keywords are actually loaded
        foundKWs = {kw:[] for kw in kws}
        for data_type in kws:
            for kw in kws[data_type]:
                if kw in self.window.DATASTRINGS:
                    foundKWs[data_type].append(kw)

        ionKw, electronKw = '', ''

        if len(foundKWs['I']) > 0:
            ionKw = foundKWs['I'][0]

        if len(foundKWs['E']) > 0:
            electronKw = foundKWs['E'][0]

        return (ionKw, electronKw)

    def getNumberDensKws(self):
        # Determine what keywords should be found for ion/electron data
        kws = {'I':[], 'E':[]}
        for sc_id in [1,2,3,4]:
            for data_type in kws:
                kw = f'MMS{sc_id}_FPI/D{data_type}S_number_density'
                kws[data_type].append(kw)

        # Determine which keywords are actually loaded
        foundKWs = {kw:[] for kw in kws}
        for data_type in kws:
            for kw in kws[data_type]:
                if kw in self.window.DATASTRINGS:
                    foundKWs[data_type].append(kw)

        ionKw, electronKw = '', ''

        if len(foundKWs['I']) > 0:
            ionKw = foundKWs['I'][0]

        if len(foundKWs['E']) > 0:
            electronKw = foundKWs['E'][0]

        return (ionKw, electronKw)

class PlaneNormalUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Plane Normal')
        Frame.resize(300, 200)
        layout = QtWidgets.QGridLayout(Frame)

        # Setup layout for user input/settings
        settingsLt = QtWidgets.QHBoxLayout()

        threshBoxLbl = QtWidgets.QLabel('  Threshold: ')
        self.threshBox = QtWidgets.QDoubleSpinBox()
        self.threshBox.setDecimals(7)
        self.threshBox.setSingleStep(0.01)

        axisBoxLbl = QtWidgets.QLabel('Reference Axis: ')
        self.axisComboBox = QtWidgets.QComboBox()
        for axis in ['X','Y','Z']:
            self.axisComboBox.addItem(axis)

        for e in [axisBoxLbl, self.axisComboBox, threshBoxLbl, self.threshBox]:
            settingsLt.addWidget(e)
        settingsLt.addStretch()

        # Set up frames for T delta and R delta
        rFrame = QtWidgets.QGroupBox('Δ R')
        tFrame = QtWidgets.QGroupBox('Δ T')
        rFrame.setAlignment(QtCore.Qt.AlignCenter)
        tFrame.setAlignment(QtCore.Qt.AlignCenter)
        rLt = QtWidgets.QHBoxLayout(rFrame)
        tLt = QtWidgets.QHBoxLayout(tFrame)

        self.rDeltaMat = MatrixWidget()
        self.tDeltaMat = VectorWidget()
        rLt.addWidget(self.rDeltaMat)
        tLt.addWidget(self.tDeltaMat)

        # Set up layout for additional information such as current state
        addInfoFrame = QtWidgets.QGroupBox('Additional Info')
        addInfoLt = QtWidgets.QGridLayout(addInfoFrame)

        threshLbl = QtWidgets.QLabel('Current Threshold: ')
        self.threshold = QtWidgets.QLabel('NaN')

        rsLbl = QtWidgets.QLabel('Reference Spacecraft: ')
        self.refSpcrft = QtWidgets.QLabel('1')

        addInfoLt.addWidget(threshLbl, 0, 0, 1, 1)
        addInfoLt.addWidget(self.threshold, 0, 1, 1, 1)
        addInfoLt.addWidget(rsLbl, 1, 0, 1, 1)
        addInfoLt.addWidget(self.refSpcrft, 1, 1, 1, 1)

        # Set up velocity and normal vector frames/layouts
        self.velLbl = QtWidgets.QLabel('NaN')
        self.velocFrame = QtWidgets.QGroupBox('Normal Velocity')
        self.velocFrame.setAlignment(QtCore.Qt.AlignCenter)
        velocLt = QtWidgets.QGridLayout(self.velocFrame)
        velocLt.addWidget(self.velLbl, 0, 0, 1, 1)

        self.vecFrame = QtWidgets.QGroupBox('Normal Vector')
        vecLt = QtWidgets.QVBoxLayout(self.vecFrame)
        self.vecFrame.setAlignment(QtCore.Qt.AlignCenter)
        self.normalVec = VectorWidget()
        vecLt.addWidget(self.normalVec)

        # Center a few layouts
        for lt in [tLt, velocLt, vecLt]:
            lt.setAlignment(QtCore.Qt.AlignCenter)

        # Add all frames/layout to main layout
        layout.addLayout(settingsLt, 0, 0, 1, 2)
        layout.addWidget(addInfoFrame, 1, 0, 1, 2)
        layout.addWidget(rFrame, 2, 0, 1, 1)
        layout.addWidget(tFrame, 2, 1, 1, 1)
        layout.addWidget(self.velocFrame, 1, 2, 1, 1)
        layout.addWidget(self.vecFrame, 2, 2, 1, 1)

        # Set up update button and status bar
        self.updateBtn = QtWidgets.QPushButton('Update')
        self.updateBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setSizeGripEnabled(False)

        # Set up button to average over a range of values
        self.avgBtn = QtWidgets.QPushButton('Average...')
        self.avgBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Arrange bottom layout
        botmLt = QtWidgets.QHBoxLayout()
        botmLt.addWidget(self.updateBtn)
        botmLt.addWidget(self.avgBtn)
        botmLt.addWidget(self.statusBar)

        layout.addLayout(botmLt, 3, 0, 1, 4)

class RangeSelectUI(object):
    def setupUI(self, Frame, maxMin, minMax):
        Frame.setWindowTitle('Threshold Range')
        Frame.resize(100, 100)
        layout = QtWidgets.QGridLayout(Frame)

        instrLbl = QtWidgets.QLabel('Please select a value range to average over: ')

        # Threshold range spinboxes set up
        rangeLt = QtWidgets.QHBoxLayout()
        minLbl = QtWidgets.QLabel('Min: ')
        maxLbl = QtWidgets.QLabel(' Max: ')
        self.threshMin = QtWidgets.QDoubleSpinBox()
        self.threshMax = QtWidgets.QDoubleSpinBox()
        for box in [self.threshMin, self.threshMax]:
            box.setDecimals(7)
            box.setMaximum(minMax)
            box.setMinimum(maxMin)
        self.threshMin.setValue(maxMin)
        self.threshMax.setValue(minMax)

        # Step size box setup
        stepLbl = QtWidgets.QLabel('Step size:')
        self.stepSize = QtWidgets.QDoubleSpinBox()
        self.stepSize.setMinimum(0.01)
        self.stepSize.setMaximum(abs(minMax - maxMin)/2)
        self.stepSize.setDecimals(3)
        self.stepSize.setValue(1)
        stepLt = QtWidgets.QHBoxLayout()
        stepLt.addWidget(stepLbl)
        stepLt.addWidget(self.stepSize)
        stepLt.addStretch(2)

        # Set up range spinboxes/labels layout
        for e in [minLbl, self.threshMin, None, maxLbl, self.threshMax, None]:
            if e == None:
                rangeLt.addStretch()
            else:
                rangeLt.addWidget(e)
        rangeLt.addStretch(2)

        self.applyBtn = QtWidgets.QPushButton('Apply')

        layout.addWidget(instrLbl, 0, 0, 1, 4)
        layout.addLayout(rangeLt, 1, 0, 1, 4)
        layout.addLayout(stepLt, 2, 0, 1, 4)
        layout.addWidget(self.applyBtn, 3, 3, 1, 1)

class RangeSelect(QtWidgets.QFrame, RangeSelectUI):
    def __init__(self, window, axis, dataRanges, maxMin, minMax, parent=None):
        super(RangeSelect, self).__init__(parent)
        self.window = window
        self.axis = axis
        self.dataRanges = dataRanges

        # Do not open anything if range not valid
        if not window.checkRangeValidity(dataRanges, maxMin, minMax):
            window.avgNormal = None
            return

        self.ui = RangeSelectUI()
        self.ui.setupUI(self, maxMin, minMax)
        self.ui.applyBtn.clicked.connect(self.apply)

    def apply(self):
        # Get parameters and set state in the main window
        step = self.ui.stepSize.value()
        startVal = self.ui.threshMin.value()
        endVal = self.ui.threshMax.value()
        self.window.avgInfo = (step, startVal, endVal)

        # Update main window w/ current range of values
        self.window.update()
        self.close()

class PlaneNormal(QtWidgets.QFrame, PlaneNormalUI, MMSTools):
    def __init__(self, window, parent=None):
        super(PlaneNormal, self).__init__(parent)

        self.lines = [] # Threshold lines shown on plots
        self.rangeSelect = None
        self.avgInfo = None
        self.lastAvgInfo = None
        MMSTools.__init__(self, window)
        self.ui = PlaneNormalUI()
        self.window = window

        self.ui.setupUI(self, window)
        self.ui.updateBtn.clicked.connect(self.update)
        self.ui.avgBtn.clicked.connect(self.openRangeSelect)
        self.ui.axisComboBox.currentIndexChanged.connect(self.update)

        self.state = 0 # Startup state, nothing has been calculated yet
        self.closed = False

    def getState(self):
        # Get axis and default threshold information
        axis = self.getAxis()
        thresh = self.getThreshold()
        avgInfo = self.lastAvgInfo # If calculating average over a range of thresholds
        return (axis, thresh, avgInfo)

    def loadState(self, state):
        axis, thresh, avgInfo = state
        self.setAxis(axis)
        self.setThreshold(thresh)
        self.avgInfo = avgInfo # Set after axis is changed (due to signals)

    def setAxis(self, axis):
        self.ui.axisComboBox.setCurrentText(axis)

    def setThreshold(self, val):
        self.ui.threshBox.setValue(val)

    def getAxis(self):
        return self.ui.axisComboBox.currentText()

    def getThreshold(self):
        return self.ui.threshBox.value()

    def openRangeSelect(self):
        # Open window to select a range to computer average over
        self.closeRangeSelect()
        axis = self.getAxis()
        dataRanges, maxMin, minMax = self.getDataRange(axis)
        self.rangeSelect = RangeSelect(self, axis, dataRanges, maxMin, minMax)
        self.rangeSelect.show()

    def closeRangeSelect(self):
        if self.rangeSelect:
            self.rangeSelect.close()
            self.rangeSelect = None

    def closeEvent(self, event):
        # Remove all threshold lines before closing
        if not self.closed:
            plotNum = 0
            for plt, line in zip(self.window.plotItems, self.lines):
                plt.removeItem(line)
                plotNum += 1
            self.window.endGeneralSelect()
            self.closeRangeSelect()
            super().closeEvent(event)

        self.closed = True

    def getDstr(self, scNum, axis):
        return self.getDstrsByVec(axis)[scNum-1]

    def detCrossoverTime(self, pt1, pt2, threshold):
        # Determine the time value between two points, using their slope
        # to determine where the threshold value would lay between them
        t1, y1 = pt1
        t2, y2 = pt2
        slope = (y2 - y1)/(t2 - t1)
        crossTime = ((threshold - y1)/slope) + t1
        return crossTime

    def detCrossTimes(self, axis, dataRanges, threshold):
        result = []
        # For each spcfrt's field data along given axis, within selected time range
        axisMap = {'X':0, 'Y':1, 'Z':2}
        for scNum, dataRange in zip([1,2,3,4], dataRanges):
            dstr = self.getDstr(scNum, axis)
            dtaStrt, dtaEnd = dataRange
            dta = self.vecArrays['Field'][scNum][axisMap[axis]]
            times = self.window.getTimes(dstr, self.window.currentEdit)[0]
            index = dtaStrt
            # Look for indices in data that threshold should fall between
            for i in range(dtaStrt, dtaEnd-1):
                d1, d2 = dta[i], dta[i+1]
                lwrBnd, upprBnd = min(d1, d2), max(d1, d2)
                if (lwrBnd <= threshold and threshold <= upprBnd):
                    index = i
                    break
            # Calculate the crossover time using both points
            pt1 = (times[index], dta[index])
            pt2 = (times[index+1], dta[index+1])
            t = self.detCrossoverTime(pt1, pt2, threshold)
            result.append(t)
        return result

    def detPos(self, pt1, pt2, t):
        # Interpolate position data at crossover time to get better estimate
        # for position time
        t1, pos1 = pt1
        t2, pos2 = pt2
        pos = np.interp([t], [t1,t2], [pos1, pos2])
        return pos[0]

    def spcrftPositions(self, crossTimes):
        # Get the position vectors for each spacecraft at their respective cross times
        spcrftPosVecs = []
        vecKeys = ['X','Y','Z']
        axisMap = {'X':0, 'Y':1, 'Z':2}
        # For every spacecraft
        for scNum in [1,2,3,4]:
            t = crossTimes[scNum-1]
            posVec = []
            # For every axis in x, y, z
            for axis in vecKeys:
                dstr = self.getDstrsByVec(axis, grp='Pos')[scNum-1]
                posDta = self.vecArrays['Pos'][scNum][axisMap[axis]]
                times = self.window.getTimes(dstr, self.window.currentEdit)[0]

                # Determine points where crossTime would fall between
                index = bisect.bisect(times, t)
                if index == 0:
                    index = 1

                # Interpolate position for given axis and append to vector list
                pt1 = (times[index-1], posDta[index-1])
                pt2 = (times[index], posDta[index])
                pos = self.detPos(pt1, pt2, t)
                posVec.append(pos)
            spcrftPosVecs.append(posVec)
        # TODO: Check for closest value, not just the first one found
        return spcrftPosVecs

    def getDataRange(self, axis):
        # Gets the bounding data indices corresponding to the time region selected
        dstrs = self.getDstrsByVec(axis)
        minVals = []
        maxVals = []
        dataRanges = []
        tO, tE = self.window.getSelectionStartEndTimes()
        axisMap = {'X':0, 'Y':1, 'Z':2}
        dtas = [self.vecArrays['Field'][scNum][axisMap[axis]] for scNum in [1,2,3,4]]
        for dstr, dta in zip(dstrs, dtas):
            # Get data indices corresp. to time selection
            times = self.window.getTimes(dstr, self.window.currentEdit)[0]
            i0 = self.window.calcDataIndexByTime(times, tO)
            i1 = self.window.calcDataIndexByTime(times, tE)
            # Get the min/max data values in the selected region
            dta = dta[i0:i1]
            minVal, maxVal = min(dta), max(dta)
            minVals.append(minVal)
            maxVals.append(maxVal)
            # Store the data indices found earlier
            dataRanges.append((i0, i1))
        maxMin = max(minVals)
        minMax = min(maxVals)

        # Returns the data ranges for each spcrft and the bounding values
        # for the range of values where they overlap
        return dataRanges, maxMin, minMax

    def checkRangeValidity(self, dataRanges, maxMin, minMax):
        # Checks if the value ranges overlap and if all spacecraft data is loaded
        valid = True
        if maxMin >= minMax:
            self.ui.statusBar.showMessage('Error: Field values do not overlap.', 5000)
            valid = False
        elif len(dataRanges) < 4:
            self.ui.statusBar.showMessage('Error: Missing spacecraft data in this range', 5000)
            valid = False
        return valid

    def inBounds(self, val, lowerBnd, upperBnd, eps=0):
        # Checks if value within range (lowerBnd, upperBnd)
        if (lowerBnd - eps) <= val and val <= (upperBnd + eps):
            return True
        else:
            return False

    def getAvgInfo(self):
        return self.avgInfo

    def update(self):
        self.waitForData()

        axis = self.getAxis()
        dataRanges, maxMin, minMax = self.getDataRange(axis)

        # If can't calculate normal vector, do nothing
        if not self.checkRangeValidity(dataRanges, maxMin, minMax):
            return

        # Update threshold box/settings if necessary
        currThresh = self.getThreshold()
        if self.state == 0 or not self.inBounds(currThresh, maxMin, minMax, 0.0005):
            threshold = (maxMin + minMax)/2
        else:
            threshold = currThresh
        self.ui.threshBox.setMinimum(maxMin)
        self.ui.threshBox.setMaximum(minMax)

        # Add horizontal lines to plots indicating threshold value
        self.addLinesToPlots(threshold)

        # Calculate for current threshold if not averaging over a range
        # of thresholds
        avgInfo = self.getAvgInfo()
        if avgInfo is None:
            rD, tD, normalVec, vel = self.calcNormal(axis, dataRanges, threshold)
        else:
            rD, tD, normalVec, vel = self.calcAvg(axis, dataRanges, avgInfo)

        # Update all UI elements with newly calculated values
        self.updateLabels(threshold, rD, tD, normalVec, vel, avgInfo)
        self.state = 1 # Mark as no longer being in startup state

        # Reset thresh range state after calculating, but keep last range info
        # in case saving workspace
        self.lastAvgInfo = self.avgInfo
        self.avgInfo = None

    def calcAvg(self, axis, dataRanges, threshInfo):
        # Extract range information and swap if not in correct order
        step, startVal, endVal = threshInfo
        startVal = min(startVal, endVal)
        endVal = max(startVal, endVal)

        # Compute normal vec/vel at each step and average
        numSteps = 0
        avgVel = 0
        avgVec = np.zeros(3)
        currVal = startVal
        while currVal <= endVal:
            rD, tD, nvec, nvel = self.calcNormal(axis, dataRanges, currVal)
            avgVec = avgVec + nvec
            avgVel = avgVel + nvel
            numSteps += 1
            currVal += step

        if numSteps == 0: # Return if failed
            return
        avgVec = avgVec / numSteps
        avgVel = avgVel / numSteps

        return rD, tD, avgVec, avgVel

    def calcNormal(self, axis, dataRanges, threshold):
        # Calculate the times where each spacecraft crosses threshold value
        crossTimes = self.detCrossTimes(axis, dataRanges, threshold)

        # Determine which spacecraft first crosses the threshold value
        firstSpcrft = 1 # Fix to MMS1 for now, might not matter

        # Get the position vectors for each spacecraft
        spcrftPosVecs = self.spcrftPositions(crossTimes)

        # Get the pos vec and cross time for the reference spacecraft
        vec0 = np.array(spcrftPosVecs[firstSpcrft-1])
        t0 = crossTimes[firstSpcrft-1]

        # Solve system of eqns of form Ax = b where A = (delta R),  b = (delta T)
        rDelta = np.zeros((3,3))
        tDelta = np.zeros(3)
        rowNum = 0
        for vec, ct, scNum in zip(spcrftPosVecs, crossTimes, [1,2,3,4]):
            if scNum == firstSpcrft:
                continue
            rDelta[rowNum] = np.array(vec) - vec0
            tDelta[rowNum] = ct - t0
            rowNum += 1
        systemSol = np.linalg.solve(rDelta, tDelta)

        # Calculate normal velocity from the solution (value that normalizes solution)
        velocity = 1 / np.sqrt(np.dot(systemSol, systemSol))

        # Multiply the system solution by the velocity to get the normal vector
        normalVec = systemSol * velocity

        return rDelta, tDelta, normalVec, velocity

    def updateLabels(self, thresh, rDelta, tDelta, normalVec, normalVel, avgInfo=None):
        prec = 5

        # Set velocity / vector label text and titles
        velTitle, vecTitle = 'Normal Velocity', 'Normal Vector'
        if avgInfo is not None: # Add avg keyword if using averaged values
            velTitle, vecTitle = 'Avg ' + velTitle, 'Avg ' + vecTitle
        self.ui.velocFrame.setTitle(velTitle)
        self.ui.vecFrame.setTitle(vecTitle)
        self.ui.normalVec.setVector(np.round(normalVec, decimals=prec))
        self.ui.velLbl.setText(str(np.round(normalVel, decimals=prec)))

        # Set threshold text
        if avgInfo is None:
            self.ui.threshold.setText(str(round(thresh, prec)))
        else:
            step, sv, ev = avgInfo
            sv, ev = np.round(sv, decimals=prec), np.round(ev, decimals=prec)
            self.ui.threshold.setText(str((sv, ev)))
        self.setThreshold(thresh) # Update value if no value set

        # Set r delta and t delta matrix labels text
        self.ui.rDeltaMat.setMatrix(np.round(rDelta, decimals=prec))
        self.ui.tDeltaMat.setVector(np.round(tDelta, decimals=prec))

    def addLinesToPlots(self, threshold):
        # Determine if any lines have previously been added to plots
        firstLines = True if self.lines == [] else False

        plotNum = 0
        for plt in self.window.plotItems:
            if firstLines:
                # Create infinite lines with y = threshold
                pen = pg.mkPen(pg.mkColor('#5d00ff'))
                infLine = pg.InfiniteLine(pos=threshold, angle=0, pen=pen)
                plt.addItem(infLine)
                self.lines.append(infLine)
            else:
                # Update line position for new thresholds
                self.lines[plotNum].setPos(threshold)
            plotNum += 1

# Calculations derived from ESA paper on curlometer technique
class CurlometerUI(BaseLayout):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Curlometer')
        Frame.resize(300, 200)
        layout = QtWidgets.QGridLayout(Frame)
        self.layout = layout

        # Set up value widgets
        self.curlB = VectorWidget(prec=5)
        self.Rmat = MatrixWidget(prec=5)
        self.AvgDensVec = VectorWidget(prec=5)
        self.divB = QtWidgets.QLabel()
        self.divBcurlB = QtWidgets.QLabel()
        self.jPar = QtWidgets.QLabel()
        self.jPerp = QtWidgets.QLabel()

        # Make value label widgets selectable
        for lbl in [self.divB, self.divBcurlB, self.jPar, self.jPerp]:
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        frameTitles = ['R', 'Curl B', '<p> Current Density J (nA/m<sup>2</sup>)</p>',
            '<p>| J<sub>PAR</sub> |</p>', '<p>| J<sub>PERP</sub> |</p>']
        frameWidgets = [self.Rmat, self.curlB, self.AvgDensVec, self.jPar, self.jPerp]
        framePositions = [(0, 0), (0, 1), (0, 2), (1, 1), (1,2)]

        for lbl, elem, pos in zip(frameTitles, frameWidgets, framePositions):
            frame, frmLt = self.getLabeledFrame(lbl)
            frmLt.addWidget(elem, 0, 0, 1, 1, QtCore.Qt.AlignCenter)
            row, col = pos
            layout.addWidget(frame, row, col, 1, 1)

        # Set up quality measures layout
        qualityFrame, qualityLt = self.getLabeledFrame('Quality Measures')
        self.addPair(qualityLt, 'Div B:', self.divB, 0, 0, 1, 1)
        self.addPair(qualityLt, '|Div B|÷|Curl B|:', self.divBcurlB, 1, 0, 1, 1)
        layout.addWidget(qualityFrame, 1, 0, 1, 1)

        # New plot variables layout setup
        prgrsLt = QtWidgets.QGridLayout()
        plotFrame = QtWidgets.QGroupBox('Generate variables for plotting: ')
        plotLt = QtWidgets.QHBoxLayout(plotFrame)
        self.jParaChk = QtWidgets.QCheckBox('| J_Para |')
        self.jMagChk = QtWidgets.QCheckBox('| J |')
        self.jPerpChk = QtWidgets.QCheckBox('| J_Perp |')
        self.JxChk = QtWidgets.QCheckBox('Jx')
        self.JyChk = QtWidgets.QCheckBox('Jy')
        self.JzChk = QtWidgets.QCheckBox('Jz')
        for i, e in enumerate([self.jMagChk, self.jParaChk, self.jPerpChk]):
            plotLt.addWidget(e)
            e.setChecked(True)
        for i, e in enumerate([self.JxChk, self.JyChk, self.JzChk]):
            plotLt.addWidget(e)
            e.setChecked(True)

        self.progressBar = QtWidgets.QProgressBar()
        self.applyBtn = QtWidgets.QPushButton('Apply')
        self.applyBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        varLt = QtWidgets.QHBoxLayout()
        varLt.addWidget(plotFrame)
        varLt.addWidget(self.applyBtn)
        varLt.addWidget(self.progressBar)
        layout.addLayout(varLt, 2, 0, 1, 3)

        # Set up TimeEdit and checkbox to keep window on top of main win
        btmLt = QtWidgets.QHBoxLayout()
        self.timeEdit = TimeEdit()
        minDt, maxDt = window.getMinAndMaxDateTime()
        self.timeEdit.setupMinMax((minDt, maxDt))
        self.onTopCheckBox = QtWidgets.QCheckBox('Stay On Top')
        self.onTopCheckBox.setChecked(True)
        btmLt.addWidget(self.timeEdit.start)
        btmLt.addWidget(self.timeEdit.end)
        btmLt.addWidget(self.onTopCheckBox)
        btmLt.addStretch()

        layout.addLayout(btmLt, 3, 0, 1, 3)
        self.glw = None

    def getLabeledFrame(self, lbl):
        wrapFrame = QtWidgets.QFrame()
        wrapLt = QtWidgets.QGridLayout(wrapFrame)
        wrapLt.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel(lbl)
        title.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        wrapLt.addWidget(title, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)

        frm = QtWidgets.QFrame()
        frm.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frmLt = QtWidgets.QGridLayout(frm)
        wrapLt.addWidget(frm, 1, 0, 1, 1)

        return wrapFrame, frmLt

class Curlometer(QtWidgets.QFrame, CurlometerUI, MMSTools):
    def __init__(self, window, parent=None):
        super(Curlometer, self).__init__(parent)
        self.window = window

        MMSTools.__init__(self, window)
        self.ui = CurlometerUI()
        self.ui.setupUI(self, window)

        # Set up window-on-top settings
        self.visibleWin = False
        self.ui.onTopCheckBox.clicked.connect(self.toggleWindowOnTop)

        # Set up new plot variables interface
        self.ui.applyBtn.clicked.connect(self.addcstmVars)
        self.ui.progressBar.setVisible(False)

        # Initializes plot variable checkboxes if previously added to DATASTRINGS
        self.nameDict = {'| J |' : 'J', '| J_Para |' : 'J_Para', 
            '| J_Perp |' : 'J_Perp', 'Jx': 'Jx', 'Jy':'Jy', 'Jz':'Jz'}
        boxes = [self.ui.jMagChk, self.ui.jParaChk, self.ui.jPerpChk,
            self.ui.JxChk, self.ui.JyChk, self.ui.JzChk]
        for box, name in zip(boxes, list(self.nameDict.values())):
            box.blockSignals(True)
            if name in window.DATASTRINGS:
                box.setChecked(True)
            box.blockSignals(False)

    def closeEvent(self, event):
        self.window.endGeneralSelect()
        self.close()

    def toggleWindowOnTop(self, val):
        # Keeps window on top of main window while user updates lines
        self.setParent(self.window if val else None)
        dialogFlag = QtCore.Qt.Dialog
        if self.window.OS == 'posix':
            dialogFlag = QtCore.Qt.Tool
        flags = self.windowFlags()
        flags = flags | dialogFlag if val else flags & ~dialogFlag
        self.setWindowFlags(flags)
        self.show()

    def calculate(self, specIndex=None):
        if not self.visibleWin:
            self.toggleWindowOnTop(self.ui.onTopCheckBox.isChecked())
            self.visibleWin = True

        refSpc = 4 # Current default for now
        mu0 = constants.mu_0

        # Determine data index corresponding to selected time
        if specIndex is not None:
            index = specIndex
            self.waitForData()
        else:
            tO, tE = self.window.getSelectionStartEndTimes()
            times = self.window.getTimes(self.window.DATASTRINGS[0], 0)[0]
            index = self.window.calcDataIndexByTime(times, tO)
            if index == len(times):
                index = index - 1

        # Calculate R, I
        R = self.getRMatrix(refSpc, index)
        I = self.getIVec(refSpc, index)

        # Calculate curl(B) and J from R and I
        Rinv = np.linalg.inv(R)
        curlB = np.matmul(Rinv, I)
        J = (1/mu0) * curlB

        # Calculate quality measures
        if specIndex is None:
            divB = self.calcDivB(refSpc, index)
            divCurlB = abs(divB) / self.getNorm(curlB)

        # Calculate J_Parallel and J_Perpendicular magnitudes
        jPara, jPerp = self.calcJperpJpar(J, index)

        # Switch all results back to original units
        jPara = self.convertToOriginal(jPara)
        jPerp = self.convertToOriginal(jPerp)
        R = self.convertToOriginal(R, dtaType='pos')
        J = self.convertToOriginal(J)

        if specIndex is None:
            divB = self.convertToOriginal(divB)
            curlB = self.convertToOriginal(curlB)

        if specIndex is None:
            self.setLabels(R, J, divB, curlB, divCurlB, (jPara, jPerp))

        return J, jPara, jPerp

    def calcDivB(self, refSpc, index):
        spcrfts = [1, 2, 3, 4]
        spcrfts.remove(refSpc)
        spcrftStack = spcrfts[:] # Object to keep track of remaining spcrfts

        i = 0
        sumVal = 0
        stackLen = len(spcrftStack)

        # Cyclic sum, use a list to cycle through all (i, j, k) orderings
        while i < stackLen:
            top = spcrftStack[0] # Current i spacecraft
            # Move to end for next combination, first two elems are now j and k
            spcrftStack.remove(top)
            spcrftStack.append(top)

            # Calculate delta B_i and delta R_j x delta R_k
            bDelta = self.getBDelta(top, refSpc, index)
            crossDiff = self.crossDiffs(spcrftStack[0], spcrftStack[1], refSpc, index)

            # Dot these two values together and add to sum
            sumVal += np.dot(bDelta, crossDiff)

            i += 1
        sumVal = sumVal

        # Calculate delta_R_i and delta_R_j x delta_R_k for first i,j,k combination
        crossDiff = self.crossDiffs(spcrfts[1], spcrfts[2], refSpc, index)
        rDelta = self.rDelta(spcrfts[0], refSpc, index)
        denom = np.dot(rDelta, crossDiff)

        # Result is the initial sum divided by this last dot product
        return abs(sumVal / denom)

    def setLabels(self, R, J, divB, curlB, divCurlB, parPerp):
        # Updates all labels in window
        self.ui.Rmat.setMatrix(R)
        self.ui.AvgDensVec.setVector(J)
        self.ui.divB.setText(np.format_float_scientific(divB, precision=5))
        self.ui.curlB.setVector(curlB)
        self.ui.divBcurlB.setText(np.format_float_scientific(divCurlB, precision=5))
        jPara, jPerp = parPerp
        self.ui.jPar.setText(str(np.round(jPara, decimals=5)))
        self.ui.jPerp.setText(str(np.round(jPerp, decimals=5)))

    def convertToSI(self, dta, dtaType='mag'):
        # Converts data units to SI units
        kmToMeters = 1e3
        ntToTesla = 1e-9
        convDta = dta
        if dtaType == 'pos':
            convDta *= kmToMeters
        elif dtaType == 'mag':
            convDta *= ntToTesla
        return convDta

    def convertToOriginal(self, dta, dtaType='mag'):
        # Converts vectors/matrices from SI units back to original units
        kmToMeters = 1e3
        ntToTesla = 1e-9
        convDta = dta
        if dtaType == 'pos':
            convDta /= kmToMeters
        elif dtaType == 'mag':
            convDta /= ntToTesla
        return convDta

    def getRMatrix(self, refSpc, index):
        # Calculates cross product for Δ r_a, Δ r_b for all (a,b) pairs
        otherSpcrfts = [1,2,3,4]
        otherSpcrfts.remove(refSpc)

        R = np.zeros((3,3))
        rowNum = 0
        pairs = [(otherSpcrfts[0], otherSpcrfts[1]),
                (otherSpcrfts[1], otherSpcrfts[2]),
                (otherSpcrfts[0], otherSpcrfts[2])]

        for i in range(0, 3):
            spc1, spc2 = pairs[rowNum]
            R[rowNum] = self.crossDiffs(spc1, spc2, refSpc, index)
            rowNum += 1

        return R

    def getIVec(self, refSpc, index):
        # Calculates Δ B_a * Δ r_b - Δ B_b * Δ r_a for all (a, b) pairs
        otherSpcrfts = [1,2,3,4]
        otherSpcrfts.remove(refSpc)

        I = np.zeros(3)
        rowNum = 0
        pairs = [(otherSpcrfts[0], otherSpcrfts[1]),
                (otherSpcrfts[1], otherSpcrfts[2]),
                (otherSpcrfts[0], otherSpcrfts[2])]

        for i in range(0, 3):
            # Calculate delta B's and delta R's
            spc1, spc2 = pairs[rowNum]
            bdelta1 = self.getBDelta(spc1, refSpc, index)
            rdelta1 = self.rDelta(spc2, refSpc, index)
            bdelta2 = self.getBDelta(spc2, refSpc, index)
            rdelta2 = self.rDelta(spc1, refSpc, index)
            # Dot and subtract
            I[rowNum] = np.dot(bdelta1, rdelta1) - np.dot(bdelta2, rdelta2)
            rowNum += 1

        return I

    def crossDiffs(self, nonRefSpc1, nonRefSpc2, refSpc, index):
        # r_ij x r_kj, where r is the rdelta vector wrt refSpc
        rDelta1 = self.rDelta(nonRefSpc1, refSpc, index)
        rDelta2 = self.rDelta(nonRefSpc2, refSpc, index)
        crossDiff = np.cross(rDelta1, rDelta2)
        return crossDiff

    def getBDelta(self, nonRefSpc, refSpc, index):
        # b_i - b_ref, where b is the magnetic field vector
        nonRefVec = self.getVec(nonRefSpc, index)
        refVec = self.getVec(refSpc, index)
        siVec = self.convertToSI(nonRefVec - refVec)
        return siVec

    def rDelta(self, nonRefSpc, refSpc, index):
        # xyz_i - xyz_ref, where xyz is the position vector
        nonRefVec = self.getVec(nonRefSpc, index, grp='Pos')
        refVec = self.getVec(refSpc, index, grp='Pos')
        siVec = self.convertToSI(nonRefVec - refVec, dtaType='pos')
        return siVec

    def calcJperpJpar(self, J, index):
        # Calculate the magnitude of J_Perp and J_Par
        B = np.zeros(3)
        for scNum in [1,2,3,4]:
            currB = self.convertToSI(np.array(self.getVec(scNum, index)))
            B += currB
        B = B / 4
        B = B / self.getNorm(B)

        jPara = np.dot(J, B)
        jProj = (np.dot(J, B) / (np.dot(B,B))) * B # J proj onto B
        jPara = abs(jPara)
        jPerp = self.getNorm(J - jProj)

        return jPara, jPerp

    def calcRange(self, indices=None, regNum=0, totRegions=1):
        # Get data indices to generate values for
        if indices is None:
            t0, t1 = self.window.minTime, self.window.maxTime
            times = self.window.getTimes(self.window.DATASTRINGS[0], 0)[0]
            i0 = 0
            i1 = len(times) - 1
        else:
            i0, i1 = indices
        numIndices = abs(i1 - i0)

        # Initialize progress bar settings/visibility
        regFrac = 100/totRegions
        progVal = regNum*regFrac
        progStep = regFrac / numIndices
        self.setProgressVis(True)

        mat = np.zeros((6, numIndices))
        for i in range(i0, i1):
            # Calculate |J|, |J_perp|, |J_par| at every index and store it
            matIndex = i - i0
            J, jPar, jPerp = self.calculate(i)
            mat[:,matIndex] = np.array([self.getNorm(J), jPar, jPerp, J[0], J[1], J[2]])
            progVal += progStep
            self.ui.progressBar.setValue(progVal)

        # Update progress bar to show finished status
        QtCore.QTimer.singleShot(2000, self.setProgressVis)

        return mat

    def lineMode(self):
        regions = self.window.currSelect.regions
        if len(regions) == 1 and regions[0].isLine():
            return True
        return False

    def getNorm(self, v):
        return np.sqrt(np.dot(v, v))

    def addcstmVars(self):
        # Calculates selected values for entire dataset and creates corresp. new vars
        varsToAdd = []
        # Gather all new variables to be added and remove any unchecked ones
        for box in [self.ui.jMagChk, self.ui.jParaChk, self.ui.jPerpChk,
            self.ui.JxChk, self.ui.JyChk, self.ui.JzChk]:
            dstr = self.nameDict[box.text()]
            if box.isChecked():
                varsToAdd.append(dstr)
            elif dstr in self.window.DATASTRINGS:
                self.window.rmvCustomVar(dstr)

        if len(varsToAdd) == 0:
            return

        regions = self.window.currSelect.regions
        refDstr = self.getDstrsBySpcrft(1)[0]
        if self.lineMode():
            # Calculate |J|, |J_Par|, |J_Perp| and initialize the new variables
            times = self.window.getTimes(refDstr, 0)
            resultMat = self.calcRange()
            for dstr in varsToAdd:
                index = list(self.nameDict.values()).index(dstr)
                self.window.initNewVar(dstr, resultMat[index], 'nA/m^2', times=times)
        elif len(regions) >= 1:
            # Get indices for each region
            indices = []
            for regNum in range(0, len(regions)):
                indexPair = self.window.calcDataIndicesFromLines(refDstr, 0, regNum)
                indices.append(indexPair)

            # Initialize time and result arrays
            times = []
            resDta = []
            for var in varsToAdd:
                resDta.append([])

            # Run calculations for each region, updating result arrays that
            # are going to be returned
            regIndex = 0
            for a, b in indices:
                times.extend(self.window.getTimes(refDstr, 0)[0][a:b])
                resultMat = self.calcRange((a,b), regIndex, len(regions))
                for dstr in varsToAdd:
                    index = list(self.nameDict.values()).index(dstr)
                    resDta[index].extend(resultMat[index])
                regIndex += 1

            # Re-calculate resolutions and create time object for the new variable
            times = np.array(times)
            prevTimes, prevRes, avgRes = self.window.getTimes(refDstr, 0)
            diff = np.diff(times)
            diff = np.concatenate([diff, [times[-1]]])
            times = (times, diff, avgRes)
            for dstr in varsToAdd:
                index = list(self.nameDict.values()).index(dstr)
                self.window.initNewVar(dstr, np.array(resDta[index]), 'nA/m^2', times=times)

    def setProgressVis(self, b=False):
        # Sets progress bar / label as hidden or visible
        self.ui.progressBar.setVisible(b)

class CurvatureUI(BaseLayout):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Curvature')
        Frame.resize(1100, 700)
        layout = QtWidgets.QGridLayout(Frame)

        # Set up settings frame
        plotFrame = self.setupPlotFrame(Frame)
        layout.addWidget(plotFrame, 0, 0, 1, 1)

        # Set up graphics grid
        self.glw = self.getGraphicsGrid()
        layout.addWidget(self.gview, 1, 0, 1, 1)

        # Set up time edit layout
        timeFrm = QtWidgets.QFrame()
        timeFrm.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
        timeLt = QtWidgets.QHBoxLayout(timeFrm)
        timeLt.setContentsMargins(0, 0, 0, 0)
        self.timeEdit = TimeEdit()
        timeLt.addWidget(self.timeEdit.start)
        timeLt.addWidget(self.timeEdit.end)
        timeLt.addStretch()

        # Hide and uncheck gyroradius buttons if missing data
        etempKw = Frame.electronKw
        itempKw = Frame.ionKw
        tempKws = [etempKw, itempKw]

        for tempKw, boxText in zip(tempKws, ['Electron Gyroradius', 'Ion Gyroradius']):
            if tempKw not in Frame.window.DATASTRINGS:
                for box in self.checkboxes:
                    if box.text() == boxText:
                        box.setChecked(False)
                        box.setVisible(False)

        layout.addWidget(timeFrm, 2, 0, 1, 1)

    def setupPlotFrame(self, Frame):
        plotFrame = QtWidgets.QGroupBox('Variables: ')
        plotLt = QtWidgets.QHBoxLayout(plotFrame)

        # Set up variable checkboxes
        self.checkboxes = []
        for name in list(Frame.nameMap.keys()):
            chkbx = QtWidgets.QCheckBox(name)
            chkbx.setChecked(True)
            self.checkboxes.append(chkbx)
            plotLt.addWidget(chkbx)

        # Set up plot / add to main / create variables buttons
        plotLt.addStretch()
        self.updateBtn = QtWidgets.QPushButton(' Plot ')
        self.addToMainBtn = QtWidgets.QPushButton(' Add To Main Window ')
        self.saveVarBtn = QtWidgets.QPushButton(' Create Variables ')
        for btn in [self.updateBtn, self.saveVarBtn, self.addToMainBtn,]:
            plotLt.addWidget(btn)

        return plotFrame

class Curvature(QtWidgets.QFrame, CurvatureUI, MMSTools):
    def __init__(self, window, parent=None):
        super(Curvature, self).__init__(parent)
        MMSTools.__init__(self, window)
        self.window = window
        self.ionKw, self.electronKw = self.getFPITempPerpKws()
        self.ui = CurvatureUI()
        self.lastCalc = None

        # Constant used in gyro-radius calculations
        self.k_b = constants.physical_constants['Boltzmann constant in eV/K'][0]

        # Maps checkbox name to keywords
        self.nameMap = {'Curvature':'Curvature', 'Radius Of Curvature':'Radius',
            'Electron Gyroradius':'RE_Gyro', 'Ion Gyroradius':'RI_Gyro'}

        # Maps variable to number of columns
        self.colMap = {'Curvature':3, 'Radius':1, 'RE_Gyro':1, 'RI_Gyro':1}

        # Maps keyword to units
        self.units = {'Curvature':'km^-1', 'Radius':'km', 'RE_Gyro':'km', 
            'RI_Gyro':'km'}

        # Maps keyword to variable names
        self.varNames = {'Curvature':['Curv_X', 'Curv_Y', 'Curv_Z'],
            'Radius':['RC'], 'RE_Gyro':['RE_Gyro'], 'RI_Gyro':['RI_Gyro']}

        # Set up plot stacked labels
        self.labels = {kw:[kw] for kw in self.colMap}
        self.labels['Curvature'] = [f'Curv<sub><b>{ax}<b></sub>' for ax in ['X','Y','Z']]
        self.labels['Radius'] = ['RC']
        self.labels['RE_Gyro'] = ['Electron Gyroradius']
        self.labels['RI_Gyro'] = ['Ion Gyroradius']

        self.ui.setupUI(self, window)
        self.eDta, self.iDta = None, None
        self.ui.updateBtn.clicked.connect(self.update)
        self.ui.saveVarBtn.clicked.connect(self.createVariables)
        self.ui.addToMainBtn.clicked.connect(self.addToMain)

    def closeEvent(self, ev):
        self.window.endGeneralSelect()
        self.close()

    def update(self):
        self.waitForData()

        # Get start/end indices and time array
        edit = 0
        arbDstr = self.getDstrsBySpcrft(1)[0]
        sI, eI = self.window.calcDataIndicesFromLines(arbDstr, edit)
        times = self.window.getTimes(arbDstr, edit)[0][sI:eI]

        # Store time array for plotting later
        self.lastCalc = {'times' : times}

        # Calculate values
        worker = TaskRunner(self.calcOverRng, sI, eI)
        worker.signals.result.connect(self.calcFinished)
        self.pool.start(worker)

    def calcFinished(self, result):
        self.lastCalc['data'] = result
        times = self.lastCalc['times']

        # Plot time series
        self.plot(result, times)

    def plot(self, results, times, copy=False):
        ''' Plots the calculated variables in the plot grid;
            Returns plots if copy is True
        '''
        # Clear and generate a new plot grid
        self.ui.glw.clear()
        pltGrd = PlotGridObject()
        pltGrd.set_window(self.window)
        self.ui.glw.addItem(pltGrd, 0, 0, 1, 1)

        # For each result
        plotItems = {}
        penIndex = 0
        index = 0
        plotColors = []
        for key in results:
            # Extract data
            data = results[key]

            # Get pens
            ncols = self.colMap[key]
            pens = self.window.pens[penIndex:penIndex + ncols]
            colors = [pen.color() for pen in pens]
            penIndex += ncols

            # Build plot items and plot each trace
            sublabels = self.labels[key]
            units = self.units[key]
            label = StackedLabel(sublabels, units=[units]*len(sublabels), colors=colors)

            plt = MagPyPlotItem(self.window.epoch)
            for col in range(0, ncols):
                plt.plot(times, data[:,col], pen=pens[col])
            plt.setXRange(times[0], times[-1], 0.0)

            # Add to grid if not just creating copy
            if not copy:
                pltGrd.get_layout().add_row()
                pltGrd.get_layout()[index] = [label, plt]
                index += 1
            else:
                plotItems[key] = (plt, label)

            # Make sure radius of curvature is log scale
            if key == 'Radius':
                plt.setLogMode(x=False, y=True)
            plotColors.append(colors)

        self.ui.glw.update()
        return plotItems, plotColors

    def createVariables(self, sigHand=None, close=True):
        ''' Creates new plot variables in main window '''
        result = self.lastCalc
        if result is None:
            return

        # Extract data from result
        dataMap, times = result['data'], result['times']
        timeInfo = (times, np.diff(times), times[1]-times[0])

        # Create variables for each key in dataMap
        for key in dataMap:
            # Get variable info for this key
            data = dataMap[key]
            cols = self.colMap[key]
            varNames = self.varNames[key]
            units = self.units[key]

            # Extract data and initalize new variable
            for col in range(0, cols):
                varDta = data[:,col]
                name = varNames[col]
                self.window.initNewVar(name, varDta, units=units, times=timeInfo)

        if close:
            self.close()

    def addToMain(self):
        ''' Adds selected plots/variables to main window's plot grid '''
        result = self.lastCalc
        if result is None:
            return

        # Generate the plot variables for the main window
        self.createVariables(close=False)

        # Create new plots from the previous results
        data, times = result['data'], result['times']
        plots, plotColors = self.plot(data, times, copy=True)

        # Add plots to main window
        for key in data:
            varLst = [(dstr, 0) for dstr in self.varNames[key]]
            self.window.lastPlotStrings.append(varLst)
        self.window.plotTracePens.extend(plotColors)
        self.window.replotGrid()

        self.close()

    def calcOverRng(self, iO, iE):
        ''' Calculates the selected variables over a given index range '''
        # Get the array sizes and number of points 
        iO, iE = min(iO, iE), max(iO, iE)
        n = iE - iO

        # Get the selected keywords and initalized arrays
        keys = [box.text() for box in self.ui.checkboxes if box.isChecked()]
        keys = [self.nameMap[key] for key in keys] # Map to internal keywords
        results = {key:np.empty((n, self.colMap[key])) for key in keys}

        # Switch to boolean values to speed things up
        baseCalc = 'Curvature' in keys or 'Radius' in keys
        radiusCalc = 'Radius' in keys
        egyroCalc = 'RE_Gyro' in keys
        igyroCalc = 'RI_Gyro' in keys

        # Calculate the values
        for i in range(iO, iE):
            refIndex = i - iO
            if baseCalc:
                curv, rc = self.calculate(i)
                results['Curvature'][refIndex] = curv
                if radiusCalc:
                    results['Radius'][refIndex] = rc

            if egyroCalc:
                results['RE_Gyro'][refIndex] = self.getGyroRadius(i, 'Electron')

            if igyroCalc:
                results['RI_Gyro'][refIndex] = self.getGyroRadius(i, 'Ion')

        # Return a dictionary of the results
        return results

    def calculate(self, index):
        # Calculates curvature, radius for given spcrft at given index
        vec = self.calcCurv(index)
        radius = self.getCurvRadius(vec)
        return vec, radius

    def getCurvRadius(self, curv):
        radius = 1 / np.sqrt(np.dot(curv, curv))
        return radius

    def interpParticleDta(self, mode='Electron'):
        # Determine temperature variable name
        tempKw = self.electronKw
        if mode == 'Ion':
            tempKw = self.ionKw

        # Create a cubic splines interpolater on ion/electron data
        # and interpolate along magnetic field data times
        tempTimes = self.window.getTimes(tempKw, 0)[0]
        tempDta = self.window.getData(tempKw, 0)
        cs = scipy.interpolate.CubicSpline(tempTimes, tempDta)
        magTimes = self.window.getTimes(self.getDstrsBySpcrft(1)[0], 0)[0]
        if mode == 'Electron': # Fill corresponding array
            self.eDta = cs(magTimes)
        else:
            self.iDta = cs(magTimes)

    def getGyroRadius(self, index, mode='Electron'):
        # Use electron omni tool to check for ion/electron data dstrs
        # and interpolate particle data before doing computations
        if self.eDta is None or self.iDta is None:
            self.iDta = []
            self.eDta = []
            if self.electronKw != '':
                self.interpParticleDta()
            if self.ionKw != '':
                self.interpParticleDta('Ion')

        # Set formula coefficients and which array to get temperature from
        if mode == 'Electron':
            coeff = 0.0221
            interpDta = self.eDta
        else:
            coeff = 0.947
            interpDta = self.iDta

        # Return nan if data not available for given mode
        if interpDta == []:
            return np.nan

        # Convert temperature from eV to kelvin
        temperature = interpDta[index]
        temperature = temperature / self.k_b

        # Compute the average magnetic field vector
        avgB = np.array(self.getVec(1, index))
        for scNum in [2,3,4]:
            avgB += self.getVec(scNum, index)
        avgB = avgB / 4
        normAvgB = np.sqrt(np.dot(avgB, avgB))

        gyroRadius = coeff * np.sqrt(temperature) / normAvgB
        return gyroRadius

    def estimateError(self, index, radius):
        # Error = (sin^(-1)(L/2R) - L/2R)/ (L/2R) --> approx = (1/6)*(L/2R)^2
        L = self.getAvgDist(index) # Average x distance between spcrft
        alpha = L / (2 * radius)
        error = (1/6) * (alpha ** 2)
        return error

    def getNorm(self, v):
        return np.sqrt(np.dot(v, v))

    def getAvgDist(self, index):
        # Finds the average x pos distance between the spacecfraft
        L = 0
        pairs = []
        for i in range(1, 4+1):
            for j in range(1, 4+1):
                if i != j and (i, j) not in pairs and (j, i) not in pairs:
                    pairs.append((i, j))

        for a, b in pairs:
            vec1 = self.getVec(a, index, grp='Pos')
            vec2 = self.getVec(b, index, grp='Pos')
            diff = self.getNorm(vec2-vec1)
            L += diff
        L = L / len(pairs)
        return L

    def getUnitVec(self, scNum, index):
        # b = B / | B |
        vec = self.getVec(scNum, index)
        magn = np.sqrt(np.dot(vec, vec))
        return vec / magn

    def calcGradient(self, index):
        bVecs = [self.getUnitVec(sc, index) for sc in [1,2,3,4]]
        rVecs = [self.getVec(sc, index, grp='Pos') for sc in [1,2,3,4]]

        # Subtract average position vector from all r Vecs
        avgVec = (rVecs[0] + rVecs[1] + rVecs[2] + rVecs[3]) / 4
        rVecs = [rv - avgVec for rv in rVecs]

        R = np.zeros((3,3))
        for k in range(0, 3):
            for j in range(0, k+1):
                r_kj = 0
                for alpha in range(0, 4):
                    r_kj += rVecs[alpha][k] * rVecs[alpha][j]
                r_kj = r_kj / 4
                R[k][j] = r_kj
                R[j][k] = r_kj

        Rinv = np.linalg.inv(R)

        G_0 = np.zeros((3, 3))
        for i in range(0, 3):
            for j in range(0, 3):
                g_0_ij = 0
                for alpha in range(0, 4):
                    for k in range(0, 3):
                        g_0_ij += bVecs[alpha][i] * rVecs[alpha][k] * Rinv[k][j]
                g_0_ij = g_0_ij / 4
                G_0[i][j] = g_0_ij

        g_0_ii = sum([G_0[i][i] for i in range(0,3)])
        rInv_ii = sum([Rinv[i][i] for i in range(0, 3)])
        lambda_i = -g_0_ii / rInv_ii

        G = np.zeros((3, 3))
        for i in range(0, 3):
            for j in range(0, 3):
                G[i][j] = G_0[i][j] + (lambda_i * Rinv[i][j])

        return G

    def getAvgBUnitVec(self, index):
        # Computes the average b vector and normalizes it
        bUnit = np.array(self.getVec(1, index))
        for scNum in [2,3,4]:
            bUnit += self.getVec(scNum, index)
        bUnit = bUnit / 4
        bUnit = bUnit / self.getNorm(bUnit)
        return bUnit

    def calcCurv(self, index, G=None):
        G = self.calcGradient(index)
        bUnit = self.getAvgBUnitVec(index)

        curvature = np.matmul(G, bUnit)
        return curvature

class MMSColorPltTool():
    def addPlotsToMain(self, kws, selectedKws, units, editLink=None):
        for plt, gradLbl, kw in zip(self.plotItems, self.gradLabels, kws):
            # Skip plots that aren't selected
            if kw not in selectedKws:
                continue

            # Pass plot to main window to add it to the main grid
            specData = plt.get_specs()[0]
            specData.set_name(kw)
            specData.set_y_label(units)
            specData.set_legend_label(gradLbl.getLabelText())

            self.window.SPECDICT[kw] = specData

        self.window.replotData()
        self.close()

    def getRangeSettings(self):
        valRngs = []
        toggles, boxes = self.ui.getRangeChksAndBoxes()
        for toggle, (minBox, maxBox) in zip(toggles, boxes):
            if toggle.isChecked():
                minVal, maxVal = minBox.value(), maxBox.value()
                valRngs.append((minVal, maxVal))
            else:
                valRngs.append(None)

        return valRngs

    def setRangeSettings(self, valRngs):
        toggles, boxes = self.ui.getRangeChksAndBoxes()
        for rng, toggle, (minBox, maxBox) in zip(valRngs, toggles, boxes):
            if rng is not None:
                minVal, maxVal = rng
                toggle.setChecked(True)
                minBox.setValue(minVal)
                maxBox.setValue(maxVal)

    def getColorScaleMode(self):
        return self.ui.scaleModeBox.currentText()

    def setColorScaleMode(self, val):
        self.ui.scaleModeBox.setCurrentText(val)

    def getState(self):
        state = {}
        state['Scale'] = self.getColorScaleMode()
        state['Ranges'] = self.getRangeSettings()
        return state

    def loadState(self, state):
        if 'Scale' in state:
            self.setColorScaleMode(state['Scale'])

        if 'Ranges' in state:
            self.setRangeSettings(state['Ranges'])

class SelectableList(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Set up input/output list widgets
        self.inputList = QtWidgets.QListWidget()
        self.outputList = QtWidgets.QListWidget()

        for lst in [self.inputList, self.outputList]:
            lst.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # # Set up general layout
        self.titleLabel = QtWidgets.QLabel()
        self.titleLabel.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.inputList)

        self.setMinimumWidth(300)

    def setLabel(self, txt):
        self.titleLabel.setText(txt)

    def setInput(self, items, datas=None):
        if datas is None:
            datas = [''] * len(items)

        for item in items:
            self.inputList.addItem(item)

    def getSelectedItems(self):
        return [item.text() for item in self.inputList.selectedItems()]

    def getInputItems(self):
        count = self.inputList.count()
        return [self.inputList.item(row).text() for row in range(count)]

    def getOutputItems(self):
        count = self.outputList.count()
        return [self.outputList.item(row).text() for row in range(count)]

    def addSelections(self):
        # Get list of new items and add to set of currently selected items
        items = self.getSelectedItems(self.inputList)
        prevItems = self.getOutputItems()
        newItems = set(items) | set(prevItems)

        # Clear output list and re-add items
        self.outputList.clear()
        for item in newItems:
            self.outputList.addItem(item)

    def removeSelections(self):
        # Get updated list of items
        items = self.getOutputItems()
        selectedItems = self.getSelectedItems(self.outputList)
        newItems = set(items) - set(selectedItems)
        newItems = sorted(list(newItems))

        # Clear items and update output list
        self.outputList.clear()
        for item in newItems:
            self.outputList.addItem(item)

class PressureToolUI(BaseLayout):
    def setupUI(self, frame):
        frame.setWindowTitle('Pressure')
        frame.resize(1200, 700)
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(0, 5, 0, 5)

        # Set up grid and default plots
        self.glw = self.getGraphicsGrid()
        self.gview.setMinimumHeight(500)
        layout.addWidget(self.gview, 0, 0, 1, 1)

        self.setupPlots(frame)

        # Set up settings layout/frame
        settingsFrm = self.setupSettingsFrm(frame)
        layout.addWidget(settingsFrm, 1, 0, 1, 1)

    def setupSettingsFrm(self, mainFrame):
        frame = QtWidgets.QFrame()
        frame.setMaximumHeight(50)
        layout = QtWidgets.QHBoxLayout(frame)

        # Set up time edits
        self.timeEdit = TimeEdit()
        layout.addWidget(self.timeEdit.start)
        layout.addWidget(self.timeEdit.end)

        layout.addStretch()

        # Set up variable checkboxes
        self.boxes = {}
        boxLt = QtWidgets.QHBoxLayout()
        for kw in mainFrame.plot_labels:
            box = QtWidgets.QCheckBox(' ' + kw.split(' ')[0])
            box.setChecked(True)
            boxLt.addWidget(box)
            self.boxes[kw] = box

            # Hide and uncheck if not in list of available kws
            if mainFrame.available_kws[kw] == False:
                box.setChecked(False)
                box.setVisible(False)

        # Adjustments for plasma beta checkbox
        self.boxes['Plasma Beta'].setText('Plasma Beta')
        self.boxes['Plasma Beta'].setChecked(False)

        layout.addLayout(boxLt)

        # Add 'Add To Main Window' button
        self.addBtn = QtWidgets.QPushButton('Add To Main Window')
        self.addBtn.setMaximumWidth(200)
        layout.addWidget(self.addBtn)

        # Add update button
        self.updtBtn = QtWidgets.QPushButton('Update')
        self.updtBtn.setMaximumWidth(100)
        layout.addWidget(self.updtBtn)

        return frame

    def setupPlots(self, frame):
        # Create plot grid and add to grid graphics layout
        grid = PlotGridObject()
        grid.set_window(frame.window)
        self.glw.addItem(grid)
        self.pltGrd = grid

        # Generate plot
        ## Set up axis items and viewbox
        plt = MagPyPlotItem(epoch=frame.window.epoch)
        plt.getAxis('bottom').enableAutoSIPrefix(False)

        ## Get plot label, trace pen, and units
        dstrs, colors = [], []
        for lbl, pen in zip(frame.plot_labels, frame.pens):
            dstrs.extend(lbl.split(' '))
            colors.extend([pen.color().name()]*2)
            units = frame.plot_info[lbl][1]

        ## Hide buttons and set top/right axes visible
        plt.hideButtons()
        for ax in ['top', 'right']:
            plt.showAxis(ax)
            plt.getAxis(ax).setStyle(showValues=False)

        ## Create stacked label and add plot + label to grid
        label = StackedLabel(dstrs, colors, units=[units]*len(dstrs))

        row = grid.get_layout().shape()[0]
        grid.get_layout().add_row()
        grid.get_layout()[row] = [label, plt]

        frame.labels.append(label)
        frame.plotItems.append(plt)

class PressureTool(QtWidgets.QFrame, PressureToolUI, MMSTools):
    def __init__(self, window, *args, **kwargs):
        super(PressureTool, self).__init__(parent=None)
        MMSTools.__init__(self, window)
        self.window = window

        # Plot item objects
        self.labels = []
        self.plotItems = []
        self.pens = window.pens[0:4]

        # Constants
        self.mu0 = constants.mu_0
        self.k_b_eV = constants.physical_constants['Boltzmann constant in eV/K'][0]
        self.k_b_j = constants.physical_constants['Boltzmann constant'][0]
        self.nt_to_T = 1e-9
        self.cm3_to_m3 = 1e6
        self.pa_to_npa = 1e9

        # Variable names
        self.i_dstr, self.e_dstr = self.getNumberDensKws()
        self.temp_kw_i, self.temp_kw_e = self.getFPITempPerpKws()

        # Plot label maps to variable name and units
        self.plot_labels = ['Magnetic Pressure', 'Thermal Pressure', 
            'Total Pressure', 'Plasma Beta']
        self.magn_lbl, self.therm_lbl, self.total_lbl = self.plot_labels[0:3]
        self.plasma_beta_lbl = self.plot_labels[3]
        self.plot_info = {
            self.plot_labels[0]: ('magn_press', 'nPa'), 
            self.plot_labels[1]: ('therm_press', 'nPa'),
            self.plot_labels[2]: ('total_press', 'nPa'),
            self.plot_labels[3]: ('plasma_beta', '')
        }

        # Groups of plot variables
        self.pltGrps =[[self.magn_lbl, self.therm_lbl, self.total_lbl], [self.plasma_beta_lbl]]

        # Determine which elements can be calculated base on
        # loaded variables
        self.available_kws = {kw:False for kw in self.plot_labels}
        all_dstrs = self.window.DATASTRINGS[:]
        mag_lbl, therm_lbl, total_lbl = self.plot_labels[0:3]
        if self.inValidState():
            self.available_kws[mag_lbl] = True
        if self.e_dstr in all_dstrs and self.i_dstr in all_dstrs:
            self.available_kws[therm_lbl] = True
            if self.available_kws[mag_lbl]:
                self.available_kws[total_lbl] = True
                self.available_kws[self.plasma_beta_lbl] = True

        # Setup UI
        self.ui = PressureToolUI()
        self.ui.setupUI(self)
        self.plotAppr = None

        # Connect buttons to functions
        self.ui.updtBtn.clicked.connect(self.update)
        self.ui.addBtn.clicked.connect(self.addToMain)

        # Stores last calculation
        self.lastCalc = None

    def calcMagneticPress(self, btotal):
        # Compute magnetic pressure as |B|^2/(2*mu0)
        btotal *= self.nt_to_T
        pressure = (btotal ** 2) / (2 * self.mu0)
        return pressure * self.pa_to_npa

    def getNumDensity(self, kw='N_Dens', index=0):
        n_dens = self.window.getData(kw, 0)[index]
        return n_dens

    def getTemperature(self, kw='TempPer', index=0):
        temp = self.window.getData(kw, 0)[index]
        return temp

    def calcThermalPress(self, kw, temp_kw, index):
        # Convert temperature in eV to Kelvin
        temp_eV = self.getTemperature(temp_kw, index=index)
        temp_K = temp_eV / self.k_b_eV

        # Get number density and convert to 1/m^3
        n_dens = self.getNumDensity(kw, index=index)
        n_dens = n_dens * self.cm3_to_m3

        # Compute thermal pressure as 
        # (number density) * (temperature in Kelvin) * (Boltzmann constant)
        therm_press = n_dens * temp_K * self.k_b_j
        return therm_press * self.pa_to_npa

    def get_indices_and_times(self, kw):
        sI, eI = self.window.calcDataIndicesFromLines(kw, 0)
        times = self.window.getTimes(kw, 0)[0][sI:eI]
        return ((sI, eI), times)

    def calcThermPressFull(self, kw, temp_kw):
        # Determines the start/end indices for the given number
        # density & tempperp keywords, and calculates the thermal
        # pressure over that range
        (sI, eI), times = self.get_indices_and_times(kw)
        therm_press = np.empty(eI-sI)

        for i in range(sI, eI):
            therm_press[i-sI] = self.calcThermalPress(kw, temp_kw, i)

        return times, therm_press

    def calcPlasmaThermalPress(self, mag_times=None):
        # Get electron thermal pressure
        times_e, therm_press_e = self.calcThermPressFull(self.e_dstr, self.temp_kw_e)

        # Get ion thermal pressure
        times_i, therm_press_i = self.calcThermPressFull(self.i_dstr, self.temp_kw_i)

        # Interpolate along mag_times if it is given, otherwise, on electron times
        ref_times = times_e if mag_times is None else mag_times
        cs = scipy.interpolate.CubicSpline(times_e, therm_press_e)
        interp_therm_press_e = cs(ref_times)
        cs = scipy.interpolate.CubicSpline(times_i, therm_press_i)
        interp_therm_press_i = cs(ref_times)

        # Remove any NaNs for extrapolated values and replace w/ endpoints
        timeLst = [times_e, times_i]
        interpLst = [interp_therm_press_e, interp_therm_press_i]
        origLst = [therm_press_e, therm_press_i]
        for t, dta, origDta in zip(timeLst, interpLst, origLst):
            # Fill starting/ending values
            dta[ref_times < t[0]] = origDta[0]
            dta[ref_times > t[-1]] = origDta[-1]

        # Sum up thermal pressure and return
        thermal_press = interp_therm_press_e + interp_therm_press_i
        return ref_times, thermal_press

    def calcMagnPressFull(self):
        # Get start/end indices and times
        mag_dstr = self.getDstrsBySpcrft(1, grp='Field')[0]
        sI, eI = self.window.calcDataIndicesFromLines(mag_dstr, 0)
        mag_times = self.window.getTimes(mag_dstr, 0)[0][sI:eI]

        # Calculate magnetic pressure over the given range
        sc = 1
        data = self.getMagData(sc, sI, eI, btot=True)
        if len(data) == 4:
            btotal = np.array(data[-1])
        else:
            btotal = np.sqrt((data[0]**2) + (data[1]**2) + (data[2]**2))
        magn_press = self.calcMagneticPress(btotal)

        return mag_times, magn_press

    def update(self):
        self.waitForData()

        times = None
        magn_press, therm_press, total_press = None, None, None
        plasma_beta = None

        # Calculate magnetic pressure
        if self.available_kws[self.magn_lbl]:
            times, magn_press = self.calcMagnPressFull()

        # Calculate plasma thermal pressure
        if self.available_kws[self.therm_lbl]:
            times, therm_press = self.calcPlasmaThermalPress(times)

        # Calculate total pressure
        if self.available_kws[self.total_lbl]:
            total_press = magn_press + therm_press

        # Calculate plasma beta
        if self.available_kws[self.plasma_beta_lbl]:
            plasma_beta = therm_press / magn_press

        # Create a dictionary from the calculated data
        results = {
            self.magn_lbl : magn_press,
            self.therm_lbl : therm_press,
            self.total_lbl : total_press,
            self.plasma_beta_lbl : plasma_beta
        }

        # Nullify any unchecked values
        for kw in results:
            if not self.ui.boxes[kw].isChecked():
                results[kw] = None

        self.lastCalc = (times, results)
        self.plotItems, self.labels = self.plotData(times, results)

    def plotData(self, times, results, copy=False):
        ''' Returns plots and labels, skips adding to grid if copy is True '''
        # Clear previous plots
        self.ui.glw.clear()
        self.plotItems = []
        self.ui.pltGrd = None

        if times is None:
            return

        # Initialize plot grids
        self.ui.pltGrd = PlotGridObject()
        self.ui.pltGrd.set_window(self.window)
        self.ui.glw.addItem(self.ui.pltGrd)

        # Use same time range for all plots
        t0, t1 = times[0], times[-1]

        # Create plot items/labels and add to grid
        plotItems = []
        labelItems = []
        penIndex = 0
        for grp in self.pltGrps:
            # Create plot item for each group
            plt = MagPyPlotItem(epoch=self.window.epoch)

            # Plot each trace in plot
            units = None
            labelList, penList = [], []
            for lbl in grp:
                # Check if variable was generated
                dta = results[lbl]
                if dta is None:
                    penIndex += 1
                    continue

                # Get pen and label info
                units = self.plot_info[lbl][1]
                if units == '':
                    units = None
                pen = self.pens[penIndex]
                penList.extend([pen]*2)
                labelList.extend(lbl.split(' '))

                # Plot trace
                plt.plot(times, dta, pen=pen, name=lbl)
                penIndex += 1

            # Skip adding empty plots
            if len(labelList) < 1:
                continue

            # Create label item
            colors = [pen.color() for pen in penList]
            labelItem = StackedLabel(labelList, colors, units=[units]*len(labelList))

            # Set plot range
            plt.setXRange(t0, t1, padding=0.0)

            # Add plot to grid
            if not copy:
                row = self.ui.pltGrd.get_layout().shape()[0]
                self.ui.pltGrd.get_layout().add_row()
                self.ui.pltGrd.get_layout()[row] = [labelItem, plt]
            plotItems.append(plt)
            labelItems.append(labelItem)

        return plotItems, labelItems

    def addToMain(self):
        self.update()

        if self.lastCalc is None:
            return

        # Get times and pressure values from last calculation
        times, results = self.lastCalc

        # If nothing plotted, do nothing except close
        if len(results) == 0:
            self.close()
            return

        # Generate the time info for each variable
        t_diff = np.diff(times)
        timeInfo = (times, t_diff, times[1] - times[0])

        # Get plot information for each plot
        penIndex = 0
        plotStrings, plotPens = [], []
        for grp in self.pltGrps:
            grpStrings, grpPens = [], []
            # Get variable information for each trace in plot
            for label in grp:
                # Check if variable has data
                dta = results[label]
                if dta is None:
                    penIndex += 1
                    continue

                # Get the variable name, units, and pen
                pen = self.pens[penIndex]
                var_name, units = self.plot_info[label]
                grpStrings.append((var_name, 0))
                grpPens.append(pen)

                # Initialize a new variable in main window
                self.window.initNewVar(var_name, dta, units=units, times=timeInfo)
                penIndex += 1

            # Skip any empty plot groups
            if len(grpStrings) > 0:
                plotStrings.append(grpStrings)
                plotPens.append(grpPens)

        # Skip if nothing plotted
        if len(plotStrings) == 0:
            self.close()
            return

        # Create label and add plots + labels to plot grid
        plts, lbls = self.plotData(times, results, copy=True)
        for plt, lbl, pltStrs, pltPens in zip(plts, lbls, plotStrings, plotPens):
            self.window.lastPlotStrings.append(pltStrs)
            self.window.plotTracePens.append(pltPens)
        self.window.replotGrid()

        # Update plot grid ranges and appearance
        self.close()

    def closeEvent(self, ev):
        self.window.endGeneralSelect()
        self.close()