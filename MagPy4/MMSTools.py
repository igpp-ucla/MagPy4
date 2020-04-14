from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from .MagPy4UI import MatrixWidget, VectorWidget, TimeEdit, NumLabel, GridGraphicsLayout, StackedLabel, PlotGrid, StackedAxisLabel

from FF_Time import FFTIME, leapFile
from .dataDisplay import DataDisplay, UTCQDate

from .dynBase import SpectrogramPlotItem, SpectraLine, SpectraLegend, SimpleColorPlot
from .pyqtgraphExtensions import MagPyColorPlot, LinkedAxis, DateAxis, MagPyPlotItem
from .selectionManager import SelectableViewBox
from .layoutTools import BaseLayout
from .plotAppearance import PressurePlotApp

import scipy
from scipy import constants

import pyqtgraph as pg
import numpy as np

import functools
import bisect
import re

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

        self.initArrays()

    def inValidState(self):
        return self.mmsState

    def raiseErrorMsg(self):
        self.window.ui.statusBar.showMessage('Error: Missing MMS Data')

    def getDstrsBySpcrft(self, scNum, grp='Field'):
        return self.scGrps[grp][scNum]

    def getDstrsByVec(self, vec, grp='Field'):
        return self.grps[grp][vec]

    def getVec(self, scNum, index, grp='Field'):
        return self.vecArrays[grp][scNum][:,index]

    def getPosData(self, scNum, startIndex=None, endIndex=None):
        if startIndex is None:
            startIndex = 0
        if endIndex is None:
            endIndex = len(self.vecArrays['Pos'][scNum][0])
        return self.vecArrays['Pos'][scNum][:,startIndex:endIndex]

    def getMagData(self, scNum, startIndex=None, endIndex=None, btot=False):
        if startIndex is None:
            startIndex = 0
        if endIndex is None:
            endIndex = len(self.vecArrays['Field'][scNum][0])
        return self.vecArrays['Field'][scNum][:,startIndex:endIndex]

    def initArrays(self):
        # Creates dictionary of arrays s.t. a column within the array
        # corresponds to a field/position vector at a given data index
        # i.e. vecArrays['Field'][4] = [[bx4 data] [by4 data] [bz4 data]]
        self.vecArrays = {}

        if self.diffMode: # Get MMS1 pos dta if in diff mode
            mms1Pos = [self.window.getData(dstr) * 6371 for dstr in self.mms1Vec]

        for grp in ['Pos', 'Field']:
            self.vecArrays[grp] = {}
            for scNum in [1,2,3,4]:
                dstrs = self.getDstrsBySpcrft(scNum, grp)
                vecArr = []
                axisIndex = 0
                for dstr in dstrs:
                    dta = self.window.getData(dstr)

                    # Add in MMS1 coordinates if scNum corresp. to difference vector
                    if self.diffMode and grp == 'Pos':
                        if scNum != 1:
                            dta = dta + mms1Pos[axisIndex]
                        else:
                            dta = mms1Pos[axisIndex]
                    vecArr.append(dta)
                    axisIndex += 1
                self.vecArrays[grp][scNum] = np.array(vecArr)

    def getDiffMode(self, posDstrs):
        # Checks if position dstrs are in difference format or direct coordinates
        # format
        posKeys = ['DX', 'DY', 'DZ']
        for dstr in posDstrs:
            for pk in posKeys:
                if pk.lower() in dstr.lower():
                    return True

        return False

    def initGroups(self):
        # Extract all position datastrings and field datastrings separately
        regPosKeys = ['X','Y','Z']
        diffPosKeys = ['DX', 'DY', 'DZ']
        posKeys = regPosKeys + diffPosKeys
        fieldKeys = ['Bx', 'By', 'Bz']
        positionDstrs = []
        fieldDstrs = []
        for dstr in self.window.DATASTRINGS:
            if dstr in self.window.cstmVars:
                continue
            for pk in posKeys:
                if pk.lower() == dstr.lower()[:len(pk)] and 'b'+pk.lower() not in dstr.lower():
                    positionDstrs.append(dstr)
            for fk in fieldKeys:
                if fk.lower() in dstr.lower():
                    fieldDstrs.append(dstr)

        # Pre-sort
        positionDstrs.sort()
        fieldDstrs.sort()

        # Organize by field vectors
        self.diffMode = self.getDiffMode(positionDstrs)
        vecKeys = ['X', 'Y', 'Z']
        posDict = {}
        fieldDict = {}
        for vk in vecKeys: # For every axis, look for matching position dstrs
            vecPosDstrs = []
            for posDstr in positionDstrs:
                if vk.lower() in posDstr.lower():
                    vecPosDstrs.append(posDstr)
            posDict[vk] = vecPosDstrs

            vecFieldDstrs = [] # Repeat for field dstrs
            for fieldDstr in fieldDstrs:
                if vk.lower() in fieldDstr.lower():
                    vecFieldDstrs.append(fieldDstr)
            fieldDict[vk] = vecFieldDstrs

        # Get MMS1 pos dstrs at end of each dictionary if in diff mode
        self.mms1Vec = []
        if self.diffMode:
            for key in posDict:
                mms1Dstr = None
                for dstr in posDict[key]:
                    if 'D'.lower() != dstr.lower()[0]:
                        self.mms1Vec.append(dstr)
                        mms1Dstr = dstr

                # Rearrange position so MMS1 dstr is in front
                if mms1Dstr:
                    if mms1Dstr in posDict[key]:
                        posDict[key].remove(mms1Dstr)
                        posDict[key] = [mms1Dstr] + posDict[key]

        self.grps['Pos'] = posDict
        self.grps['Field'] = fieldDict

        # Check state here before proceeding
        posLens = [len(v) for k, v in posDict.items()]
        fieldLens = [len(v) for k, v in fieldDict.items()]
        minPL, maxPL = min(posLens), max(posLens)
        minFL, maxFL = min(fieldLens), max(fieldLens)
        if minPL != 4 or maxPL != 4 or minFL != 4 or maxFL != 4:
            self.mmsState = False
            return

        # Organize by spacecraft number
        spcrftNums = [1, 2, 3, 4]
        scPosDict = {}
        scFieldDict = {}
        for spcrftNum in spcrftNums:
            scPosDstrs = [] # For every spacecraft, get its position dstrs
            for pos in posDict.keys():
                scPosDstrs.append(posDict[pos][spcrftNum-1])
            scPosDict[spcrftNum] = scPosDstrs

            scFieldDstrs = [] # Repeat for its field dstrs
            for field in fieldDict.keys():
                scFieldDstrs.append(fieldDict[field][spcrftNum-1])
            scFieldDict[spcrftNum] = scFieldDstrs

        self.scGrps['Pos'] = scPosDict
        self.scGrps['Field'] = scFieldDict

        # Identify and B_TOTAL keywords
        for dstr in self.window.DATASTRINGS:
            if dstr.startswith('BT'):
                self.btotDstrs.append(dstr)

        self.btotDstrs.sort()

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

class RangeSelect(QtGui.QFrame, RangeSelectUI):
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

class PlaneNormal(QtGui.QFrame, PlaneNormalUI, MMSTools):
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
        plotNum = 0
        for plt, line in zip(self.window.plotItems, self.lines):
            plt.removeItem(line)
            plotNum += 1

        self.window.endGeneralSelect()
        self.closeRangeSelect()
        self.close()

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
        self.timeEdit = TimeEdit(QtGui.QFont())
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

class Curlometer(QtGui.QFrame, CurlometerUI, MMSTools):
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

class CurvatureUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Curvature Results')
        Frame.resize(100, 100)
        layout = QtWidgets.QGridLayout(Frame)

        # Set up curvature, radius, and error frames
        self.vecs = []
        curvFrame = self.setupMMSFrame('Curvature', self.vecs, VectorWidget, 5)

        self.radii = []
        radiiFrame = self.setupMMSFrame('Radius of Curvature', self.radii, NumLabel, 6)

        self.errors = []
        errFrame = self.setupMMSFrame('Estimated Error', self.errors, NumLabel, 6)

        # Set up plot frame
        plotFrame = self.setupPlotFrame()

        gyroLt = self.setupGyroLt()
        avgDistFrame = self.setupAvgDistFrame()

        # Add everything to main layout
        layout.addWidget(curvFrame, 0, 0, 2, 1)
        layout.addWidget(radiiFrame, 0, 1, 1, 1)
        layout.addWidget(errFrame, 1, 1, 1, 1)
        layout.addWidget(plotFrame, 3, 0, 1, 3)
        layout.addLayout(gyroLt, 0, 2, 2, 1)
        layout.addWidget(avgDistFrame, 0, 3, 1, 1)

        # Set up settings layout for time edit, progress bar, and checkbox
        self.timeEdit = TimeEdit(QtGui.QFont())
        minDt, maxDt = window.getMinAndMaxDateTime()
        self.timeEdit.setupMinMax((minDt, maxDt))
        self.stayOnTopChk = QtWidgets.QCheckBox('Stay on top')
        self.stayOnTopChk.setChecked(True)

        self.progBar = QtWidgets.QProgressBar()
        self.progBar.setVisible(False)

        settingsLt = QtWidgets.QHBoxLayout()
        for e in [self.timeEdit.start, self.timeEdit.end, self.progBar, self.stayOnTopChk]:
            settingsLt.addWidget(e)

        layout.addLayout(settingsLt, 4, 0, 1, 2)

    def setupMMSFrame(self, name, lst, widgetType, prec):
        frame = QtWidgets.QGroupBox(name)
        frameLt = QtWidgets.QGridLayout(frame)
        frame.setAlignment(QtCore.Qt.AlignCenter)
        frame.setContentsMargins(0, 18, 0, 2)

        # Creates a separate groupbox frame for each element in main frame
        for i in range(0, 1):
            widget = widgetType(prec=prec)
            frameLt.addWidget(widget, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
            lst.append(widget)

        return frame

    def setupPlotFrame(self):
        plotFrame = QtWidgets.QGroupBox('Generate plot variables: ')
        plotLt = QtWidgets.QHBoxLayout(plotFrame)

        self.checkboxes = []
        for name in ['Curv_X', 'Curv_Y', 'Curv_Z', 'Radius', 'RE_Gyro', 'RI_Gyro']:
            chkbx = QtWidgets.QCheckBox(name)
            chkbx.setChecked(True)
            self.checkboxes.append(chkbx)
            plotLt.addWidget(chkbx)

        self.applyBtn = QtWidgets.QPushButton('Apply')
        plotLt.addWidget(self.applyBtn)
        return plotFrame

    def setupAvgDistFrame(self):
        avgDistFrame = QtWidgets.QGroupBox('Avg Distance')
        avgDistLt = QtWidgets.QGridLayout(avgDistFrame)
        self.avgDistLbl = NumLabel(prec=4)
        avgDistLt.addWidget(self.avgDistLbl, 0, 0, 1, 1, QtCore.Qt.AlignCenter)
        avgDistFrame.setAlignment(QtCore.Qt.AlignCenter)
        return avgDistFrame

    def setupGyroLt(self):
        # Set up average distance and gyro radius frames
        infoLt = QtWidgets.QVBoxLayout()
        alignCenter = QtCore.Qt.AlignCenter

        electGyroFrame = QtWidgets.QGroupBox('Electron Gyro Radius')
        gyroRadiusLt = QtWidgets.QGridLayout(electGyroFrame)
        self.eGyroRadiusLbl = NumLabel(prec=4)
        gyroRadiusLt.addWidget(self.eGyroRadiusLbl, 0, 0, 1, 1, alignCenter)

        ionGyroFrame = QtWidgets.QGroupBox('Ion Gyro Radius')
        gyroRadiusLt = QtWidgets.QGridLayout(ionGyroFrame)
        self.iGyroRadiusLbl = NumLabel(prec=4)
        gyroRadiusLt.addWidget(self.iGyroRadiusLbl, 0, 0, 1, 1, alignCenter)

        # Center align frames and add to layout
        for frm in [electGyroFrame, ionGyroFrame]:
            frm.setAlignment(QtCore.Qt.AlignCenter)
            infoLt.addWidget(frm)

        return infoLt

class Curvature(QtGui.QFrame, CurvatureUI, MMSTools):
    def __init__(self, window, parent=None):
        super(Curvature, self).__init__(parent)
        MMSTools.__init__(self, window)
        self.window = window
        self.ui = CurvatureUI()
        self.ui.setupUI(self, window)

        # Constant used in gyro-radius calculations
        self.k_b = constants.physical_constants['Boltzmann constant in eV/K'][0]
        self.eDta, self.iDta = None, None

        self.ui.stayOnTopChk.clicked.connect(self.toggleWindowOnTop)
        self.visibleWin = False
        self.ui.applyBtn.clicked.connect(self.addcstmVars)

    def closeEvent(self, ev):
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

    def updateCalculations(self):
        if not self.visibleWin:
            self.visibleWin = True
            self.toggleWindowOnTop(self.ui.stayOnTopChk.isChecked())

        # Determine selected data index
        tO, tE = self.window.getSelectionStartEndTimes()
        dstr = self.getDstrsBySpcrft(1)[0] # Use any relevant dstr
        times = self.window.getTimes(dstr, self.window.currentEdit)[0]
        index = self.window.calcDataIndexByTime(times, tO)

        # Run calculations for given spacecraft at this index and update window
        vec, radius, err = self.calculate(index)
        self.ui.vecs[0].setVector(vec)
        self.ui.radii[0].setText(radius)
        self.ui.errors[0].setText(err)

        # Calculate additional state information
        avgDist = self.getAvgDist(index)
        self.ui.avgDistLbl.setText(avgDist)

        eGyroRadius = self.getGyroRadius(index)
        self.ui.eGyroRadiusLbl.setText(eGyroRadius)

        iGyroRadius = self.getGyroRadius(index, 'Ion')
        self.ui.iGyroRadiusLbl.setText(iGyroRadius)

    def calculate(self, index):
        # Calculates curvature, radius, & error for given spcrft at given index
        vec = self.calcCurv(index)
        radius = self.getCurvRadius(vec)
        err = self.estimateError(index, radius)
        return vec, radius, err

    def getCurvRadius(self, curv):
        radius = 1 / np.sqrt(np.dot(curv, curv))
        return radius

    def interpParticleDta(self, mode='Electron'):
        # Determine temperature variable name
        tempKw = 'TempPer'
        if mode == 'Ion':
            tempKw = tempKw + '_I'

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
            electOmni = ElectronOmni(self.window)
            self.iDta = []
            self.eDta = []
            if electOmni.electronDstrs != []:
                self.interpParticleDta()
            if electOmni.ionDstrs != []:
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

    def addcstmVars(self):
        dstrs = ['Curv_X', 'Curv_Y', 'Curv_Z'] # Prefixes for variable names

        # Determine the number of data points to use
        arbDstr = self.getDstrsBySpcrft(1)[0]
        dta = self.window.getData(arbDstr)
        timeInfo = self.window.getTimes(arbDstr, 0)
        dtaLen = len(dta)

        # Prepare progress bar for updating
        self.ui.progBar.setVisible(True)
        stepSize = 100 / dtaLen
        progVal = 0

        plotRadii = self.ui.checkboxes[len(dstrs)].isChecked()
        plotEGyro = self.ui.checkboxes[len(dstrs)+1].isChecked()
        plotIGyro = self.ui.checkboxes[len(dstrs)+2].isChecked()
        plotCurves = True in [self.ui.checkboxes[i].isChecked() for i in [0,1,2]]

        regions = self.window.currSelect.regions
        refDstr = self.getDstrsBySpcrft(1)[0]
        indices = []
        if len(regions) == 1 and regions[0].isLine():
            indices = [(0, dtaLen)] # Full data set if no region selected
        elif len(regions) >= 1:
            # Get indices for each selected region
            indices = []
            for regNum in range(0, len(regions)):
                indexPair = self.window.calcDataIndicesFromLines(refDstr, 0, regNum)
                indices.append(indexPair)

            # Set up new times array/object from selected regions
            oldTimes, oldRes, avgRes = self.window.getTimes(refDstr, 0)
            times = []
            numIndices = 0
            for a, b in indices:
                times.extend(self.window.getTimes(refDstr, 0)[0][a:b])
                numIndices += (b-a)
            times = np.array(times)
            diff = np.diff(times)
            diff = np.concatenate([diff, [times[-1]]])
            timeInfo = (times, diff, avgRes)

            # Adjust the step size based on the number of points selected
            stepSize = 100 / numIndices

        # Create a mask of the selected indices to use to extract data from
        # results arrays
        mask = [np.arange(a,b) for (a,b) in indices]
        mask = np.concatenate(mask)
    
        # Calculate curvatures and radii if checked
        curvatures = np.zeros((3, dtaLen))
        radii = np.zeros(dtaLen)
        eGyroRadii = np.zeros(dtaLen)
        iGyroRadii = np.zeros(dtaLen)
        for a, b in indices: # Calculate individually for each region
            for index in range(a, b):
                if plotCurves or plotRadii:
                    vec, radius, err = self.calculate(index)
                    curvatures[:,index] = vec
                if plotRadii:
                    radii[index] = radius
                if plotEGyro:
                    eGyroRadii[index] = self.getGyroRadius(index, 'Electron')
                if plotIGyro:
                    iGyroRadii[index] = self.getGyroRadius(index, 'Ion')

                # Update progress bar
                progVal += stepSize
                self.ui.progBar.setValue(progVal)

        curvatures = np.array([curvatures[row][mask] for row in range(0, 3)])
        radii = radii[mask]
        eGyroRadii = eGyroRadii[mask]
        iGyroRadii = iGyroRadii[mask]

        # Create new variables
        i = 0
        for dstr in dstrs:
            if self.ui.checkboxes[i].isChecked():
                self.window.initNewVar(dstr, curvatures[i], '1/km', times=timeInfo)
            i += 1
        if plotRadii:
            self.window.initNewVar('RC', radii, 'km', times=timeInfo)
        # Add in gyro radius calculations if valid
        if plotEGyro and self.eDta != []:
            self.window.initNewVar('RE_Gyro', eGyroRadii, 'km', times=timeInfo)
        if plotIGyro and self.iDta != []:
            self.window.initNewVar('RI_Gyro', iGyroRadii, 'km', times=timeInfo)

        # Hide progress bar after work is done
        self.ui.progBar.setVisible(False)

class ParticlePlotItem(SpectrogramPlotItem):
    def __init__(self, epoch, logMode=False):
        SpectrogramPlotItem.__init__(self, epoch, logMode)

    def plotSetup(self):
        SpectrogramPlotItem.plotSetup(self)
        # Additional plot appearance adjustments specific to EPAD plots
        self.getAxis('bottom').setStyle(tickLength=4)
        self.getAxis('bottom').setStyle(tickTextOffset=2)
        self.getAxis('left').setStyle(tickLength=4)
        self.getViewBox().setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

class PitchAnglePlotItem(ParticlePlotItem):
    def __init__(self, epoch, logMode=False):
        ParticlePlotItem.__init__(self, epoch, logMode)

    def plotSetup(self):
        ParticlePlotItem.plotSetup(self)
        self.getAxis('left').setCstmTickSpacing(30)

class ScientificSpinBox(QtWidgets.QDoubleSpinBox):
    def validate(self, txt, pos):
        # Checks if string matches scientific notation format or is a regular num
        state = 1
        if re.fullmatch('\d*\.*\d*', txt):
            state = 2
        elif re.fullmatch('\d*.*\d*e\+\d+', txt) is not None:
            state = 2
        elif re.match('\d+.*\d*e', txt) or re.match('\d+.*\d*e\+', txt):
            state = 1
        else:
            state = 0

        # Case where prefix is set to '10^'
        if self.prefix() == '10^':
            if re.match('10\^\d*\.*\d*', txt) or re.match('10\^-\d*\.*\d*', txt):
                state = 2
            elif re.match('10\^\d*\.', txt) or re.match('10\^-\d*\.', txt):
                state = 1

        return (state, txt, pos)

    def textFromValue(self, value):
        if value >= 10000:
            return np.format_float_scientific(value, precision=4, trim='-', 
                pad_left=False, sign=False)
        else:
            return str(value)

    def valueFromText(self, text):
        if '10^' in text:
            text = text.split('^')[1]
        return float(text)

class ElectronPitchAngleUI(BaseLayout):
    def setupUI(self, Frame, window):
        maxSizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        Frame.setWindowTitle('Electron Pitch-Angle Distribution')
        Frame.resize(1200, 800)
        layout = QtWidgets.QGridLayout(Frame)

        # Set up plot grid
        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout(window)
        self.gview.setCentralItem(self.glw)
        self.glw.layout.setHorizontalSpacing(10)
        self.glw.layout.setContentsMargins(15, 10, 25, 10)

        # Set up time edits and status bar
        self.timeEdit = TimeEdit(QtGui.QFont())
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())
        self.statusBar = QtWidgets.QStatusBar()
        timeLt = QtWidgets.QHBoxLayout()
        timeLt.addWidget(self.timeEdit.start)
        timeLt.addWidget(self.timeEdit.end)
        timeLt.addWidget(self.statusBar)
        timeLt.addStretch()

        self.rangeElems = []
        settingsLt = self.setupSettingsLt()
        self.addPltBtn = QtWidgets.QPushButton('Add To Main Grid')

        # Add everything to main layout
        layout.addLayout(settingsLt, 0, 1, 1, 1)
        layout.addWidget(self.gview, 0, 0, 1, 1)
        layout.addLayout(timeLt, 1, 0, 1, 2)

    def getVerticalSpacer(self, ht=10):
        spacer = QtWidgets.QSpacerItem(0, ht, QSizePolicy.Maximum, QSizePolicy.Minimum)
        return spacer

    def setupSettingsLt(self):
        settingsLt = QtWidgets.QVBoxLayout()
        frame = QtWidgets.QGroupBox(' Plot Settings')
        layout = QtWidgets.QVBoxLayout(frame)

        # Set up color map scaling mode box/layout
        scaleModeLbl = QtWidgets.QLabel('Color Map Scale:')
        self.scaleModeBox = QtWidgets.QComboBox()
        self.scaleModeBox.addItem('Logarithmic')
        self.scaleModeBox.addItem('Linear')
        scaleLt = QtWidgets.QVBoxLayout()
        scaleLt.addWidget(scaleModeLbl)
        scaleLt.addWidget(self.scaleModeBox)
        layout.addLayout(scaleLt)

        spacer = self.getVerticalSpacer()
        layout.addItem(spacer)

        # Set up range settings boxes for each plot
        num = 0
        self.rangeToggles = []
        lbls = ['High Energy', 'Mid Energy', 'Low Energy']
        for lbl in lbls:
            # Create groupbox w/ lbl as title
            frm = QtWidgets.QGroupBox(lbl)
            subLt = QtWidgets.QVBoxLayout(frm)
            subLt.setContentsMargins(5,5,5,5)
            layout.addWidget(frm)
            layout.addItem(self.getVerticalSpacer())

            # Set up, store, and add range sublayout/elements to groupbox
            rngLt, selectToggle, rngElems = self.getRangeLt()
            selectToggle.toggled.connect(functools.partial(self.valRngSelectToggled, num))
            self.rangeElems.append(rngElems)
            self.rangeToggles.append(selectToggle)
            self.valRngSelectToggled(num, False)
            subLt.addLayout(rngLt)
            num += 1

        # Set up default spinbox min/max settings
        self.colorScaleToggled()

        # Add in update button
        settingsLt.addWidget(frame)
        layout.addItem(self.getVerticalSpacer(2))
        self.updtBtn = QtWidgets.QPushButton('Update')
        self.updtBtn.setFixedWidth(150)
        updtLt = QtWidgets.QHBoxLayout()
        updtLt.addWidget(self.updtBtn)

        layout.addLayout(updtLt)
        layout.addItem(self.getVerticalSpacer(5))

        settingsLt.addStretch()
        selectLt, self.addToPlotBtn, self.addCheckboxes = self.setupAddToWinLt()
        settingsLt.addLayout(selectLt)

        return settingsLt

    def setupAddToWinLt(self):
        addLt = QtWidgets.QVBoxLayout()
        btn = QtWidgets.QPushButton('Add to Main Window')
        selectionBox = QtWidgets.QGroupBox('Selected Plots')
        selectionBox.setAlignment(QtCore.Qt.AlignCenter)
        selectionBox.setToolTip('Plots that will be added to main window')
        selectionLt = QtWidgets.QHBoxLayout(selectionBox)
        checkBoxes = []
        for kw in ['High', 'Mid', 'Lo']:
            checkBox = QtWidgets.QCheckBox(kw)
            checkBox.setChecked(True)
            selectionLt.addWidget(checkBox)
            checkBoxes.append(checkBox)
        addLt.addWidget(selectionBox)
        addLt.addWidget(btn)
        return addLt, btn, checkBoxes

    def getRangeLt(self):
        selectToggle = QtWidgets.QCheckBox(' Set Value Range: ')
        rangeLt = QtWidgets.QGridLayout()

        rngTip = 'Toggle to set max/min values represented by color gradient'
        selectToggle.setToolTip(rngTip)

        minTip = 'Minimum value represented by color gradient'
        maxTip = 'Maximum value represented by color gradient'

        valueMin = ScientificSpinBox()
        valueMax = ScientificSpinBox()

        # Set spinbox defaults
        for box in [valueMax, valueMin]:
            box.setFixedWidth(100)
            box.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        spc = '       ' # Spaces that keep spinbox lbls aligned w/ chkbx lbl

        rangeLt.addWidget(selectToggle, 0, 0, 1, 2)
        maxLbl = self.addPair(rangeLt, spc+'Max: ', valueMax, 1, 0, 1, 1, maxTip)
        minLbl = self.addPair(rangeLt, spc+'Min: ', valueMin, 2, 0, 1, 1, minTip)

        # Connects checkbox to func that enables/disables rangeLt's items
        return rangeLt, selectToggle, (valueMin, valueMax, minLbl, maxLbl)

    def addPair(self, layout, name, elem, row, col, rowspan, colspan, tooltip=None):
        # Create a label for given widget and place both into layout
        lbl = QtWidgets.QLabel(name)
        lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        if name != '':
            layout.addWidget(lbl, row, col, 1, 1)
        layout.addWidget(elem, row, col+1, rowspan, colspan)

        # Set any tooltips if given
        if tooltip is not None:
            lbl.setToolTip(tooltip)

        return lbl

    def valRngSelectToggled(self, num, val):
        # Enables/disables range settings layout if toggled
        for elem in self.rangeElems[num]:
            elem.setEnabled(val)

    def colorScaleToggled(self):
        # Update spinboxes when color map scale is changed
        logMode = self.isLogColorScale()
        if logMode:
            minVal, maxVal = -100, 100
            prefix = '10^'
        else:
            minVal, maxVal = 0, 1e32
            prefix = ''

        for rngElems in self.rangeElems:
            for box in rngElems[0:2]:
                box.setMinimum(minVal)
                box.setMaximum(maxVal)
                box.setPrefix(prefix)

        for i in range(0, len(self.rangeToggles)):
            self.rangeToggles[i].setChecked(False)

    def isLogColorScale(self):
        if self.scaleModeBox.currentText() == 'Logarithmic':
            return True
        else:
            return False

    def getRangeChksAndBoxes(self):
        toggles = self.rangeToggles
        boxes = [elems[0:2] for elems in self.rangeElems]
        return toggles, boxes

class MMSColorPltTool():
    def inMainWindow(self, kw):
        kwFound = False
        pltIndex = None
        for lbl in self.window.pltGrd.labels:
            if kw in lbl.dstrs:
                kwFound = True
                pltIndex = self.window.pltGrd.labels.index(lbl)
                break
        return kwFound, pltIndex

    def addPlotsToMain(self, kws, selectedKws, units, editLink=None):
        links = []
        for plt, gradLbl, kw in zip(self.plotItems, self.gradLabels, kws):
            # Skip plots that aren't selected
            if kw not in selectedKws:
                continue

            # Pass plot to main window to add it to the main grid
            pltIndex = self.window.addColorPltToGrid(plt, kw, gradLbl, units)
            links.append(pltIndex)

            # Update ranges and resize
            self.window.updateXRange()

        # Update plot links and close current window
        self.window.lastPlotLinks.append(links)
        self.window.pltGrd.resizeEvent(None)
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

class ElectronPitchAngle(QtGui.QFrame, ElectronPitchAngleUI, MMSColorPltTool):
    def __init__(self, window, parent=None):
        super(ElectronPitchAngle, self).__init__(parent)
        MMSColorPltTool.__init__(self)

        self.ui = ElectronPitchAngleUI()
        self.window = window
        self.wasClosed = False

        self.lowKw, self.midKw, self.hiKw = 'PAD_Lo', 'PAD_Mid', 'PAD_Hi'
        self.paDstrs = self.findStrings()
        self.plotItems = []

        self.ui.setupUI(self, window)
        self.ui.updtBtn.clicked.connect(self.update)
        self.ui.scaleModeBox.currentIndexChanged.connect(self.ui.colorScaleToggled)
        self.ui.addToPlotBtn.clicked.connect(self.addToMainWindow)

        # Valid state checking
        paDstrsLens = list(map(len, self.paDstrs))
        if min(paDstrsLens) != 30 and max(paDstrsLens) != 30:
            self.window.ui.statusBar.showMessage('Error: Missing particle data')
            return

    def addToMainWindow(self):
        kws = ['E'+kw for kw in [self.hiKw, self.midKw, self.lowKw]]
        selectedKws = []
        for kw in kws: # Gather all selected keywords
            index = kws.index(kw)
            if self.ui.addCheckboxes[index].isChecked():
                selectedKws.append(kw)

        self.addPlotsToMain(kws, selectedKws, 'Degrees', None)

    def findStrings(self):
        # Extract the variable names corresponding to each electron pitch-angle
        # distribution set (each dstr corresponds to one row of data)
        lowDstrs, midDstrs, hiDstrs = [], [], []

        for dstr in self.window.DATASTRINGS:
            if self.lowKw in dstr:
                lowDstrs.append(dstr)
            elif self.midKw in dstr:
                midDstrs.append(dstr)
            elif self.hiKw in dstr:
                hiDstrs.append(dstr)

        return [lowDstrs, midDstrs, hiDstrs]

    def buildValGrids(self, paDstrs, dtaRange):
        # Builds grids from the data for each pitch-angle set (lo, mid, hi)
        # and returns a list of these grids
        valueGrids = []
        for dstrLst in paDstrs:
            subGrid = []
            for dstr in dstrLst:
                # Extract data only from the selected range
                rowDta = self.window.getData(dstr)[dtaRange[0]:dtaRange[1]]
                subGrid.append(rowDta)
            subGrid = np.array(subGrid)
            valueGrids.append(subGrid)

        return valueGrids

    def update(self):
        # Get variable names and selected data range
        dataRng = self.window.calcDataIndicesFromLines(self.paDstrs[0][0], 0)

        # Clear previous plots and add title
        self.ui.glw.clear()
        title = pg.LabelItem('Electron Pitch-Angle Distribution')
        self.ui.glw.addItem(title, 0, 0, 1, 4)

        # Generate the value grids for each plot
        labels = ['Electron '+lbl+' Energy PAD' for lbl in ['High', 'Mid', 'Low']]
        pixelGrids = self.buildValGrids(self.paDstrs, dataRng)
        pixelGrids.reverse() # Plot in reverse order so high energy is top-most plot
        yVals = [i for i in range(0, 180+1, 6)] # Pitch angle values

        # Extract the selected time ticks and subtract offset
        times = self.window.getTimes(self.paDstrs[0][0], 0)[0]
        times = np.array(times[dataRng[0]:dataRng[1]])
        times = np.append(times, times[-1])

        self.gradients = []
        self.plotItems = []
        self.gradLabels = []
        logColor = self.ui.isLogColorScale()

        # Create each plot from its value grid and any user parameters
        pltNum = 0
        index = 0
        valRngs = self.getRangeSettings()
        for pixelGrid, lbl in zip(pixelGrids, labels):
            plt = PitchAnglePlotItem(self.window.epoch)

            # Check if custom value range is set
            index = labels.index(lbl)
            selectToggle = valRngs[index]
            if selectToggle is not None: # User selected range
                minVal, maxVal = selectToggle
                if logColor: # Map log scale values to non-log values
                    minVal = 10 ** minVal
                    maxVal = 10 ** maxVal
            else:
                # Otherwise, use min/max values in grid
                minVal = np.min(pixelGrid)
                maxVal = np.max(pixelGrid)
                if logColor: # Ignore zeros if color mapping scale is in log mode
                    minVal = np.min(pixelGrid[pixelGrid>0])
                    maxVal = np.max(pixelGrid[pixelGrid>0])

                # Initialize min/max values for value range spinboxes
                minBox, maxBox = self.ui.rangeElems[index][0:2]
                minBox.setValue(minVal)
                maxBox.setValue(maxVal)
                if logColor:
                    minBox.setValue(np.log10(minVal))
                    maxBox.setValue(np.log10(maxVal))
            colorRng = (minVal, maxVal)

            # Generate the color mapped plot
            plt.createPlot(yVals, pixelGrid, times, colorRng, logColor)
            self.plotItems.append(plt)
            self.ui.glw.addItem(plt, pltNum + 1, 1, 1, 1)

            # Add in the y axis label
            lbl = StackedAxisLabel([lbl, '[Degrees]'], angle=-90)
            lbl.setFixedWidth(40)
            self.ui.glw.addItem(lbl, pltNum + 1, 0, 1, 1)

            # Update date time axis and set plot view ranges
            plt.setXRange(times[0], times[-1], 0.0)
            plt.setYRange(yVals[0], yVals[-1], 0.0)

            # Add in gradient color bar
            grad = plt.getGradLegend(logColor, (1, 26))
            gradWidth = 45 if logColor else 65
            grad.setFixedWidth(gradWidth)
            grad.setEdgeMargins(0, 0)
            grad.setBarWidth(28)
            self.ui.glw.addItem(grad, pltNum + 1, 2, 1, 1)
            self.gradients.append(grad)

            # Add in color bar label
            unitsLbl = 'Log DEF' if logColor else 'DEF'
            lbl = StackedAxisLabel([unitsLbl, '[keV/(cm^2 s sr keV)]'])
            lbl.setFixedWidth(40)
            self.ui.glw.addItem(lbl, pltNum + 1, 3, 1, 1)
            self.gradLabels.append(lbl)

            # Set bottom axis defaults
            plt.getAxis('bottom').setStyle(showValues=True)
            plt.getAxis('bottom').showLabel(False)

            pltNum += 1

        # Set time label for bottom axis and adjust margins for gradient bar
        plt = self.plotItems[-1]
        grad = self.gradients[-1]
        grad.setOffsets(1, 45, 0, 0)
        plt.getAxis('bottom').showLabel(True)

        # Add in time and file info labels
        lbl = self.getTimeRangeLbl(times[0], times[-1])
        self.ui.glw.addItem(lbl, pltNum + 1, 0, 1, 3)

        self.ui.statusBar.clearMessage()

    def getTimeRangeLbl(self, t1, t2):
        t1Str = self.window.getTimestampFromTick(t1)
        t2Str = self.window.getTimestampFromTick(t2)
        txt = 'Time Range: ' + t1Str + ' to ' + t2Str
        lbl = pg.LabelItem(txt)
        lbl.setAttr('justify', 'left')
        return lbl

    def closeEvent(self, ev):
        self.ui.glw.clear()
        self.window.endGeneralSelect()
        for plt in self.plotItems:
            plt.closePlotAppearance()
        self.wasClosed = True
        self.close()

class ElectronOmniUI(BaseLayout):
    def setupUI(self, Frame, window, edta=[], idta=[]):
        layout = QtWidgets.QGridLayout(Frame)
        self.glw = self.getGraphicsGrid(window)
        self.edta = edta
        self.idta = idta

        # Set window size
        if edta == [] or idta == []:
            Frame.resize(1100, 300) # Single plot
        else:
            Frame.resize(1100, 600) # Two plots

        if edta == []:
            Frame.setWindowTitle('Omni-directional Electron Energy Spectrum')
        elif idta == []:
            Frame.setWindowTitle('Omni-directional Ion Energy Spectrum')
        else:
            Frame.setWindowTitle('Omni-directional Electron/Ion Energy Spectrum')

        settingsLt = self.setupSettingsLt()
        layout.addLayout(settingsLt, 0, 1, 1, 1)

        layout.addWidget(self.gview, 0, 0, 1, 1) # Graphics/plot grid

        timeLt, self.timeEdit, self.statusBar = self.getTimeStatusBar()
        layout.addLayout(timeLt, 1, 0, 1, 2)

    def setupSettingsLt(self):
        sideBarLt = QtWidgets.QVBoxLayout()
        settingsFrame = QtWidgets.QGroupBox('Settings')
        layout = QtWidgets.QVBoxLayout(settingsFrame)

        # Set up color scaling settings UI
        scaleLbl = QtWidgets.QLabel('Color Scaling Mode:')
        self.scaleModeBox = QtWidgets.QComboBox()
        self.scaleModeBox.addItems(['Logarithmic', 'Linear'])
        self.scaleModeBox.currentTextChanged.connect(self.colorScaleToggled)
        colorLt = QtWidgets.QVBoxLayout()
        colorLt.addWidget(scaleLbl)
        colorLt.addWidget(self.scaleModeBox)
        layout.addLayout(colorLt)

        # Set up min/max boxes for each item + toggles
        self.valBoxes = {'Electron':[], 'Ion':[]}
        self.valToggles = {}

        for dta, name in [(self.edta, 'Electron'), (self.idta, 'Ion')]:
            if dta == []:
                continue
            valueFrame = QtWidgets.QGroupBox(' Set Value Range: ')
            valueFrame.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
            valueFrame.setCheckable(True)
            self.valToggles[name] = valueFrame
            valueLt = QtWidgets.QGridLayout(valueFrame)
            for lbl, row in [('  Max: ', 0), ('  Min: ', 2)]:
                box = ScientificSpinBox()
                box.setFixedWidth(125)
                self.valBoxes[name].append(box)
                self.addPair(valueLt, lbl, box, row, 0, 1, 1)
            layout.addWidget(valueFrame)

        self.colorScaleToggled(self.scaleModeBox.currentText())

        # Set up update button
        self.updtBtn = QtWidgets.QPushButton('Update')
        layout.addWidget(self.updtBtn)
        sideBarLt.addWidget(settingsFrame)

        # Add 'add to main' btn
        sideBarLt.addStretch()
        self.addToMainBtn = QtWidgets.QPushButton('Add To Main Window')
        sideBarLt.addWidget(self.addToMainBtn)
        return sideBarLt
    
    def getRangeChksAndBoxes(self):
        toggles = []
        boxes = []
        for kw in ['Electron', 'Ion']:
            if kw in self.valBoxes:
                toggles.append(self.valToggles[kw])
                boxes.append(self.valBoxes[kw])
        
        return toggles, boxes

    def colorScaleToggled(self, val):
        # Reset selection boxes
        for dstr, valToggle in self.valToggles.items():
            valToggle.setChecked(False)

        # Adjust min/max spinboxes values according to color scaling mode
        for dstr, boxes in self.valBoxes.items():
            for box in boxes:
                if val == 'Logarithmic':
                    box.setPrefix('10^')
                    box.setMinimum(-100)
                    box.setMaximum(100)
                else:
                    box.setPrefix('')
                    box.setMinimum(1e-24)
                    box.setMaximum(1e24)

class ElectronOmni(QtWidgets.QFrame, ElectronOmniUI, MMSColorPltTool):
    def __init__(self, window, parent=None):
        super(ElectronOmni, self).__init__(parent)
        MMSColorPltTool.__init__(self)
        self.ui = ElectronOmniUI()
        self.window = window
        self.wasClosed = False

        # State and parameter information
        self.plotItems = []
        self.kwString = 'En_Omn'
        self.electronDstrs, self.ionDstrs = self.findStrings()

        # Default energy bin values
        arr1 = np.array([1.51000e+00, 1.51000e+00, 2.37000e+00, 2.62000e+00, 3.79000e+00,
            4.44000e+00, 6.14000e+00, 7.45000e+00, 1.00100e+01, 1.24200e+01,
            1.63900e+01, 2.06200e+01, 2.69100e+01, 3.41400e+01, 4.42800e+01,
            5.64400e+01, 7.29300e+01, 9.32500e+01, 1.20210e+02, 1.53970e+02,
            1.98210e+02, 2.54160e+02, 3.26900e+02, 4.19470e+02, 5.39230e+02,
            6.92200e+02, 8.89550e+02, 1.14218e+03, 1.46754e+03, 1.88460e+03,
            2.42117e+03, 3.10952e+03])
        arr2 = np.array([1.71000e+00, 1.71000e+00, 2.69000e+00, 2.97000e+00, 4.30000e+00,
            5.04000e+00, 6.96000e+00, 8.45000e+00, 1.13400e+01, 1.40800e+01,
            1.85700e+01, 2.33600e+01, 3.05000e+01, 3.86900e+01, 5.01800e+01,
            6.39700e+01, 8.26600e+01, 1.05680e+02, 1.36240e+02, 1.74510e+02,
            2.24640e+02, 2.88060e+02, 3.70490e+02, 4.75400e+02, 6.11130e+02,
            7.84500e+02, 1.00817e+03, 1.29449e+03, 1.66324e+03, 2.13591e+03,
            2.74403e+03, 3.52417e+03])

        self.energyBins = [np.mean([arr1[i], arr2[i]]) for i in range(0, 32)]
        self.energyBins.sort()

        # Set up ui and link buttons to actions
        self.ui.setupUI(self, window, self.electronDstrs, self.ionDstrs)
        self.ui.updtBtn.clicked.connect(self.update)
        self.ui.addToMainBtn.clicked.connect(self.addToMain)

    def findStrings(self):
        electDstrs = []
        ionDstrs = []
        for dstr in self.window.DATASTRINGS:
            if self.kwString in dstr:
                if '_I' in dstr: # Gather ion spectrum variable names
                    ionDstrs.append(dstr)
                else: # Gather electron spectrum variable names
                    electDstrs.append(dstr)
        return electDstrs, ionDstrs

    def addToMain(self):
        kws = ['Electron Spectrum', 'Ion Spectrum']
        selectedKws = []
        if self.electronDstrs != []:
            selectedKws.append(kws[0])
        if self.ionDstrs != []:
            selectedKws.append(kws[1])

        self.addPlotsToMain(selectedKws, selectedKws, 'eV', None)

    def update(self):
        self.plotItems = []
        self.gradLabels = []
        self.gradients = []
        self.ui.glw.clear()

        rowNum = 0
        for dstrLst, pltName in zip([self.electronDstrs, self.ionDstrs],
                ['Electron', 'Ion']):
            if dstrLst == []: # Skip either missing data set
                continue
            # Generate color map plot elements, add them to grid, and store
            plt, grad, lbl = self.plotData(mode=pltName)
            self.ui.glw.addItem(plt, rowNum, 0, 1, 1)
            self.ui.glw.addItem(grad, rowNum, 1, 1, 1)
            self.ui.glw.addItem(lbl, rowNum, 2, 1, 1)
            self.plotItems.append(plt)
            self.gradients.append(grad)
            self.gradLabels.append(lbl)
            rowNum += 1

    def plotData(self, mode='Electron'):
        dstrs = self.electronDstrs if mode == 'Electron' else self.ionDstrs
        i0, i1 = self.window.calcDataIndicesFromLines(dstrs[0], self.window.currentEdit)
        yVals = self.energyBins

        # Build grid from data slices for each variable in list
        times = self.window.getTimes(dstrs[0], self.window.currentEdit)[0]
        times = times[i0:i1+1] # Use inclusive endpoint for times
        valGrid = []
        for dstr in dstrs:
            dta = self.window.getData(dstr, self.window.currentEdit)
            valGrid.append(dta[i0:i1])
        valGrid = np.array(valGrid)

        # Determine min/max ranges for color scale
        logColor = True if self.getColorScaleMode() == 'Logarithmic' else False
        maxBox, minBox = self.ui.valBoxes[mode]
        setToggle = self.ui.valToggles[mode]
        if setToggle.isChecked(): # Use user-set values
            minVal = minBox.value()
            maxVal = maxBox.value()
            if logColor:
                minVal = 10 ** minVal
                maxVal = 10 ** maxVal
        else: # Get min/max from grid and set default values in spinboxes
            if logColor:
                # Get valid min/max only for log mode
                minVal = np.min(valGrid[valGrid>0])
                maxVal = np.max(valGrid[valGrid>0])
                minBox.setValue(np.log10(minVal))
                maxBox.setValue(np.log10(maxVal))
            else:
                minVal = np.min(valGrid)
                maxVal = np.max(valGrid)
                minBox.setValue(minVal)
                maxBox.setValue(maxVal)

        # Adjust frequency lower bound
        diff = yVals[1] - yVals[0]
        yVals = [yVals[0] - diff] + yVals

        # Create color-mapped plot
        plt = ParticlePlotItem(self.window.epoch, True)
        plt.createPlot(yVals, valGrid, times, (minVal, maxVal), logColorScale=logColor)

        title = 'Omni-directional ' + mode + ' Energy Spectrum'

        # Set plot title and update time ticks/labels
        plt.setTitle(title)

        # Create gradient object
        grad = plt.getGradLegend(logMode=logColor, offsets=(31, 45))
        grad.setMaximumWidth(50 if logColor else 85)
        grad.setBarWidth(28)

        # Add in units label
        unitsLbl = 'Log DEF' if logColor else 'DEF'
        lbl = StackedAxisLabel([unitsLbl, '[keV/(cm^2 s sr keV)]'])
        lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))

        return plt, grad, lbl

    def closeEvent(self, ev):
        self.window.endGeneralSelect()
        for plt in self.plotItems:
            plt.closePlotAppearance()
        self.wasClosed = True
        self.close()

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
        self.timeEdit = TimeEdit(QtGui.QFont())
        layout.addWidget(self.timeEdit.start)
        layout.addWidget(self.timeEdit.end)

        layout.addStretch()

        # Set up variable checkboxes
        self.boxes = []
        boxLt = QtWidgets.QHBoxLayout()
        for kw in mainFrame.plot_labels:
            box = QtWidgets.QCheckBox(' ' + kw.split(' ')[0])
            box.setChecked(True)
            boxLt.addWidget(box)
            self.boxes.append(box)

            # Hide and uncheck if not in list of available kws
            if mainFrame.available_kws[kw] == False:
                box.setChecked(False)
                box.setVisible(False)

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
        grid = PlotGrid()
        self.glw.addItem(grid)
        self.pltGrd = grid

        # Create plot appearance menu action
        apprAct = QtWidgets.QAction('Plot Appearance...')
        apprAct.triggered.connect(frame.openPlotAppr)

        # Generate plot
        ## Set up axis items and viewbox
        la = LinkedAxis('left')
        ra = LinkedAxis('right')
        ba = DateAxis(frame.window.epoch, 'bottom')
        ta = DateAxis(frame.window.epoch, 'top')
        vb = SelectableViewBox(frame.window, 0)
        vb.addMenuAction(apprAct)
        ba.enableAutoSIPrefix(False)

        plt = MagPyPlotItem(viewBox=vb, axisItems={'left':la, 'bottom':ba, 
            'right':ra, 'top':ta})

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
        label = StackedLabel(dstrs, colors, units=units)
        grid.addPlt(plt, label)

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
        self.pens = window.pens[0:3]

        # Constants
        self.mu0 = constants.mu_0
        self.k_b_eV = constants.physical_constants['Boltzmann constant in eV/K'][0]
        self.k_b_j = constants.physical_constants['Boltzmann constant'][0]
        self.nt_to_T = 1e-9
        self.cm3_to_m3 = 1e6
        self.pa_to_npa = 1e9

        # Variable names
        self.e_dstr = 'N_Dens' 
        self.temp_kw_e = 'TempPer'
        self.i_dstr = 'N_Dens_I'
        self.temp_kw_i = 'TempPer_I'

        # Plot label maps to variable name and units
        self.plot_labels = ['Magnetic Pressure', 'Thermal Pressure', 'Total Pressure']
        self.magn_lbl, self.therm_lbl, self.total_lbl = self.plot_labels
        self.plot_info = {
            self.plot_labels[0]: ('magn_press', 'nPa'), 
            self.plot_labels[1]: ('therm_press', 'nPa'),
            self.plot_labels[2]: ('total_press', 'nPa')
        }

        # Determine which elements can be calculated base on
        # loaded variables
        self.available_kws = {kw:False for kw in self.plot_labels}
        all_dstrs = self.window.DATASTRINGS[:]
        mag_lbl, therm_lbl, total_lbl = self.plot_labels
        if self.inValidState():
            self.available_kws[mag_lbl] = True
        if self.e_dstr in all_dstrs and self.i_dstr in all_dstrs:
            self.available_kws[therm_lbl] = True
            if self.available_kws[mag_lbl]:
                self.available_kws[total_lbl] = True

        # Setup UI
        self.ui = PressureToolUI()
        self.ui.setupUI(self)
        self.plotAppr = None

        # Connect buttons to functions
        self.ui.updtBtn.clicked.connect(self.update)
        self.ui.addBtn.clicked.connect(self.addToMain)

        # Stores last calculation
        self.lastCalc = None

    def calcMagneticPress(self, scNum, index):
        # Compute magnetic pressure as |B|^2/(2*mu0)
        btot_dstr = self.btotDstrs[scNum-1]
        btot_val = self.window.getData(btot_dstr, 0)[index]
        btot_val *= self.nt_to_T

        pressure = (btot_val ** 2) / (2 * self.mu0)
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
        cs = scipy.interpolate.CubicSpline(times_e, therm_press_e, extrapolate=True)
        interp_therm_press_e = cs(ref_times)
        cs = scipy.interpolate.CubicSpline(times_i, therm_press_i, extrapolate=True)
        interp_therm_press_i = cs(ref_times)

        # Sum up thermal pressure and return
        thermal_press = interp_therm_press_e + interp_therm_press_i
        return ref_times, thermal_press

    def calcMagnPressFull(self):
        # Get start/end indices and times
        mag_dstr = self.getDstrsBySpcrft(1, grp='Field')[0]
        sI, eI = self.window.calcDataIndicesFromLines(mag_dstr, 0)
        mag_times = self.window.getTimes(mag_dstr, 0)[0][sI:eI]

        # Calculate magnetic pressure over the given range
        magn_press = np.empty(eI - sI)
        for i in range(sI, eI):
            magn_press[i-sI] = self.calcMagneticPress(1, i)

        return mag_times, magn_press

    def update(self):
        times = None
        magn_press, therm_press, total_press = None, None, None

        # Calculate magnetic pressure
        if self.available_kws[self.magn_lbl]:
            times, magn_press = self.calcMagnPressFull()

        # Calculate plasma thermal pressure
        if self.available_kws[self.therm_lbl]:
            times, therm_press = self.calcPlasmaThermalPress(times)

        # Calculate total pressure
        if self.available_kws[self.total_lbl]:
            total_press = magn_press + therm_press

        # Plot calculated data
        pressures = [magn_press, therm_press, total_press]
        self.lastCalc = (times, pressures)
        self.plotData(times, pressures)

    def plotData(self, times, pressures):
        plt = self.plotItems[0]
        plt.clear()

        if times is None:
            plt.clear()
            return

        # Use same time range for all plots
        t0, t1 = times[0], times[-1]

        # Clear plot and plot each time v. pressure trace
        lbls = self.plot_labels
        checks = self.ui.boxes

        checked_kws = []
        checked_colors = []
        for dta, pen, lbl, chk in zip(pressures, self.pens, lbls, checks):
            if chk.isChecked() and dta is not None:
                plt.plot(times, dta, pen=pen, name=lbl)
                checked_kws.extend(lbl.split(' '))
                checked_colors.extend([pen.color().name()]*2)

        plt.setXRange(t0, t1, padding=0.0)
        self.set_label(checked_kws, checked_colors)

    def openPlotAppr(self):
        self.closePlotAppr()
        self.plotAppr = PressurePlotApp(self, self.plotItems, links=[[0]])
        self.plotAppr.show()

    def closePlotAppr(self):
        if self.plotAppr:
            self.plotAppr.close()
            self.plotAppr = None
    
    def set_label(self, kws, colors):
        lbl = StackedLabel(kws, colors, units='nPa')
        self.ui.pltGrd.setPlotLabel(lbl, 0)
        self.labels[0] = lbl

    def addToMain(self):
        self.update()

        if self.lastCalc is None:
            return

        # Get times and pressure values from last calculation
        times, pressures = self.lastCalc

        # If nothing plotted, do nothing except close
        if len(pressures) == 0:
            self.close()
            return

        # Get plot item and label
        plt = self.plotItems[0]
        lbl = self.labels[0]

        # Create a new plot to add
        oldPlt = plt
        vb = SelectableViewBox(None, 0)
        axes = {
            'left': LinkedAxis('left'),
            'bottom': DateAxis(self.window.epoch, 'bottom'),
            'top': DateAxis(self.window.epoch, 'top')
            }
        plt = MagPyPlotItem(viewBox=vb, axisItems=axes)
        for pdi in oldPlt.listDataItems():
            oldPlt.removeItem(pdi)
            plt.addItem(pdi)

        self.ui.pltGrd.clear()

        # Generate the time info for each variable
        t_diff = np.diff(times)
        timeInfo = (times, t_diff, times[1] - times[0])

        # For each trace, keep in plot if it is checked
        pens = self.pens
        dstrs, colors = [], []
        plotStrings = []
        plotPens = []
        for kw, pen, dta, check in zip(self.plot_labels, pens, pressures, 
            self.ui.boxes):

            if not check.isChecked() or dta is None:
                continue

            # Get the variable name, units, and pen color
            var_name, units = self.plot_info[kw]
            plotStrings.append((var_name, 0))
            colors.extend([pen.color()]*2)
            dstrs.extend(kw.split(' '))
            plotPens.append(pen)

            # Initialize a new variable in main window
            self.window.initNewVar(var_name, dta, units=units, times=timeInfo)

        if len(plotStrings) == 0:
            self.close()
            return

        # Create label and add plot + label to plot grid
        self.window.addPlot(plt, lbl, plotStrings, pens=plotPens)

        # Update plot grid ranges and appearance
        self.window.updateXRange()
        self.window.pltGrd.resizeEvent(None)
        self.close()

    def closeEvent(self, ev):
        self.closePlotAppr()
        self.window.endGeneralSelect()
        self.close()