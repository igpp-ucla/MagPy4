from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from MagPy4UI import MatrixWidget, VectorWidget

import pyqtgraph as pg
import numpy as np

import functools
import bisect

class MMSTools():
    def __init__(self, window):
        self.window = window
        # Dicts mapping field/pos kws to datastrings
        self.grps = {} # Internally maps by field axis (bx1, bx2, ...)
        self.scGrps = {} # Internally maps by spacecraft number (bx1, by1, ...)
        self.initGroups()

    def getDstrsBySpcrft(self, scNum, grp='Field'):
        return self.scGrps[grp][scNum]

    def getDstrsByVec(self, vec, grp='Field'):
        return self.grps[grp][vec]

    def initGroups(self):
        # Extract all position datastrings and field datastrings separately
        posKeys = ['X','Y','Z']
        fieldKeys = ['Bx', 'By', 'Bz']
        positionDstrs = []
        fieldDstrs = []
        for dstr in self.window.DATASTRINGS:
            for pk in posKeys:
                if pk.lower() in dstr.lower() and 'b'+pk.lower() not in dstr.lower():
                    positionDstrs.append(dstr)
            for fk in fieldKeys:
                if fk.lower() in dstr.lower():
                    fieldDstrs.append(dstr)

        # Pre-sort
        positionDstrs.sort()
        fieldDstrs.sort()

        # Organize by field vectors
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

        self.grps['Pos'] = posDict
        self.grps['Field'] = fieldDict

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
        velocFrame = QtWidgets.QGroupBox('Normal Velocity')
        velocFrame.setAlignment(QtCore.Qt.AlignCenter)
        velocLt = QtWidgets.QGridLayout(velocFrame)
        velocLt.addWidget(self.velLbl, 0, 0, 1, 1)

        vecFrame = QtWidgets.QGroupBox('Normal Vector')
        vecLt = QtWidgets.QVBoxLayout(vecFrame)
        vecFrame.setAlignment(QtCore.Qt.AlignCenter)
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
        layout.addWidget(velocFrame, 1, 2, 1, 1)
        layout.addWidget(vecFrame, 2, 2, 1, 1)

        # Set up update button and status bar
        self.updateBtn = QtWidgets.QPushButton('Update')
        self.updateBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setSizeGripEnabled(False)

        layout.addWidget(self.updateBtn, 3, 0, 1, 1)
        layout.addWidget(self.statusBar, 3, 1, 1, 3)

class PlaneNormal(QtGui.QFrame, PlaneNormalUI, MMSTools):
    def __init__(self, window, parent=None):
        super(PlaneNormal, self).__init__(parent)

        MMSTools.__init__(self, window)
        self.ui = PlaneNormalUI()
        self.window = window
        self.lines = [] # Threshold lines shown on plots

        self.ui.setupUI(self, window)
        self.ui.updateBtn.clicked.connect(self.calculate)
        self.ui.axisComboBox.currentIndexChanged.connect(self.calculate)
        self.state = 0 # Startup state, nothing has been calculated yet

    def closeEvent(self, event):
        # Remove all threshold lines before closing
        plotNum = 0
        for plt, line in zip(self.window.plotItems, self.lines):
            plt.removeItem(line)
            plotNum += 1

        self.window.endGeneralSelect()
        self.close()

    def getVec(self, scNum, index):
        # Gets the spacecraft's data vector (bx, by, bz) at a given data index
        dstrs = self.getDstrsBySpcrft(scNum)
        vec = []
        for dstr in dstrs:
            dta = self.window.getData(dstr)
            vec.append(dta[index])
        return np.array(vec)

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
        for scNum, dataRange in zip([1,2,3,4], dataRanges):
            dstr = self.getDstr(scNum, axis)
            dtaStrt, dtaEnd = dataRange
            dta = self.window.getData(dstr)
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
        # For every spacecraft
        for scNum in [1,2,3,4]:
            t = crossTimes[scNum-1]
            posVec = []
            # For every axis in x, y, z
            for axis in vecKeys:
                dstr = self.getDstrsByVec(axis, grp='Pos')[scNum-1]
                posDta = self.window.getData(dstr)
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
        for dstr in dstrs:
            # Get data indices corresp. to time selection
            times = self.window.getTimes(dstr, self.window.currentEdit)[0]
            i0 = self.window.calcDataIndexByTime(times, tO)
            i1 = self.window.calcDataIndexByTime(times, tE)
            # Get the min/max data values in the selected region
            dta = self.window.getData(dstr)
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

    def calculate(self):
        axis = self.ui.axisComboBox.currentText()
        dataRanges, maxMin, minMax = self.getDataRange(axis)

        # If can't calculate normal vector, do nothing
        if not self.checkRangeValidity(dataRanges, maxMin, minMax):
            return

        # Update threshold box/settings if necessary
        currThresh = self.ui.threshBox.value()
        if self.state == 0 or not self.inBounds(currThresh, maxMin, minMax, 0.0005):
            threshold = (maxMin + minMax)/2
        else:
            threshold = currThresh
        self.ui.threshBox.setMinimum(maxMin)
        self.ui.threshBox.setMaximum(minMax)

        # Add horizontal lines to plots indicating threshold value
        self.addLinesToPlots(threshold)

        # Calculate the normal vector and update window
        self.determineNormal(axis, dataRanges, threshold)
        self.state = 1 # Mark as no longer being in startup state

    def determineNormal(self, axis, dataRanges, threshold):
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

        # Update all UI elements with newly calculated values
        self.updateLabels(threshold, rDelta, tDelta, normalVec, velocity)

    def updateLabels(self, threshold, rDelta, tDelta, normalVec, normalVeloc):
        prec = 5
        self.ui.threshold.setText(str(round(threshold, prec)))
        self.ui.threshBox.setValue(threshold)
        self.ui.rDeltaMat.setMatrix(np.round(rDelta, decimals=prec))
        self.ui.tDeltaMat.setVector(np.round(tDelta, decimals=prec))
        self.ui.normalVec.setVector(np.round(normalVec, decimals=prec))
        self.ui.velLbl.setText(str(np.round(normalVeloc, decimals=prec)))

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