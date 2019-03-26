from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from MagPy4UI import MatrixWidget, VectorWidget, TimeEdit
from dataDisplay import DataDisplay, UTCQDate
from FF_Time import FFTIME, leapFile

from scipy import constants

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

    def getVec(self, scNum, index, grp='Field'):
        # Gets the spacecraft's data vector (bx, by, bz) at a given data index
        dstrs = self.getDstrsBySpcrft(scNum, grp)
        vec = []
        for dstr in dstrs:
            dta = self.window.getData(dstr)
            vec.append(dta[index])
        return np.array(vec)

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
        self.ui.applyBtn.clicked.connect(self.calcAvg)

    def calcAvg(self):
        # Get parameters
        step = self.ui.stepSize.value()
        startVal = self.ui.threshMin.value()
        endVal = self.ui.threshMax.value()
        # Swap if not in correct order
        startVal = min(startVal, endVal)
        endVal = max(startVal, endVal)

        # Compute normal vec/vel at each step and average
        numSteps = 0
        avgVel = 0
        avgVec = np.zeros(3)
        currVal = startVal
        while currVal <= endVal:
            rD, tD, nvec, nvel = self.window.calcNormal(self.axis, self.dataRanges, currVal)
            avgVec = avgVec + nvec
            avgVel = avgVel + nvel
            numSteps += 1
            currVal += step

        if numSteps == 0: # Return if failed
            return
        avgVec = avgVec / numSteps
        avgVel = avgVel / numSteps

        # Update main window's displayed values and close this window
        self.updateUI(avgVec, avgVel, startVal, endVal)
        self.close()

    def updateUI(self, vec, vel, startVal, endVal):
        # Change frame titles and values + updt current threshold text w/ range
        prec = 5
        self.window.ui.velocFrame.setTitle('Avg Normal Velocity')
        self.window.ui.velLbl.setText(str(np.round(vel, decimals=prec)))
        self.window.ui.vecFrame.setTitle('Avg Normal Vector')
        self.window.ui.normalVec.setVector(np.round(vec, decimals=prec))
        sv, ev = np.round(startVal, decimals=prec), np.round(endVal, decimals=prec)
        self.window.ui.threshold.setText(str((sv, ev)))

class PlaneNormal(QtGui.QFrame, PlaneNormalUI, MMSTools):
    def __init__(self, window, parent=None):
        super(PlaneNormal, self).__init__(parent)

        MMSTools.__init__(self, window)
        self.ui = PlaneNormalUI()
        self.window = window
        self.lines = [] # Threshold lines shown on plots

        self.ui.setupUI(self, window)
        self.ui.updateBtn.clicked.connect(self.calculate)
        self.ui.avgBtn.clicked.connect(self.openRangeSelect)
        self.ui.axisComboBox.currentIndexChanged.connect(self.calculate)

        self.state = 0 # Startup state, nothing has been calculated yet
        self.rangeSelect = None

    def openRangeSelect(self):
        # Open window to select a range to computer average over
        self.closeRangeSelect()
        axis = self.ui.axisComboBox.currentText()
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
        rD, tD, normalVec, vel = self.calcNormal(axis, dataRanges, threshold)

        # Update all UI elements with newly calculated values
        self.updateLabels(threshold, rD, tD, normalVec, vel)
        self.state = 1 # Mark as no longer being in startup state

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

    def updateLabels(self, threshold, rDelta, tDelta, normalVec, normalVeloc):
        prec = 5
        self.ui.velocFrame.setTitle('Normal Velocity')
        self.ui.vecFrame.setTitle('Normal Vector')
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

# Calculations derived from ESA paper on curlometer technique
class CurlometerUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Curlometer')
        Frame.resize(300, 200)
        layout = QtWidgets.QGridLayout(Frame)

        # Set up frames for values w/ own frames
        rFrame = QtWidgets.QFrame()
        JFrame = QtWidgets.QFrame()
        curlFrame = QtWidgets.QFrame()
        jParFrame = QtWidgets.QFrame()
        jPerpFrame = QtWidgets.QFrame()
        frames = [rFrame, JFrame, curlFrame, jParFrame, jPerpFrame]
        for frm in frames:
            frm.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        # Set up layouts for centering widgets
        rLt = QtWidgets.QHBoxLayout(rFrame)
        jLt = QtWidgets.QHBoxLayout(JFrame)
        curlLt = QtWidgets.QHBoxLayout(curlFrame)
        jParLt = QtWidgets.QHBoxLayout(jParFrame)
        jPerpLt = QtWidgets.QHBoxLayout(jPerpFrame)
        layouts = [rLt, jLt, curlLt, jParLt, jPerpLt]

        # Set up value widgets
        self.curlB = VectorWidget(prec=5)
        self.Rmat = MatrixWidget(prec=5)
        self.AvgDensVec = VectorWidget(prec=5)
        self.divB = QtWidgets.QLabel()
        self.divBcurlB = QtWidgets.QLabel()
        self.jPar = QtWidgets.QLabel()
        self.jPerp = QtWidgets.QLabel()
        widgets = [self.Rmat, self.AvgDensVec, self.curlB, self.jPar, self.jPerp]

        # Make value label widgets selectable
        for lbl in [self.divB, self.divBcurlB, self.jPar, self.jPerp]:
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # Center layouts that contain only one element
        for lt, wdg in zip(layouts, widgets):
            lt.addStretch()
            lt.addWidget(wdg)
            lt.addStretch()

        # Set up labels for all widgets that will display values
        rLbl = QtWidgets.QLabel('R')
        jLbl = QtWidgets.QLabel('<p> Current Density J (nA/m<sup>2</sup>)</p>')
        divLbl = QtWidgets.QLabel('<p>Div B: </p>')
        curlLbl = QtWidgets.QLabel('<p>Curl B</p>')
        ratioLbl = QtWidgets.QLabel('<p>|Div B|÷|Curl B|:</p>')
        jParLbl = QtWidgets.QLabel('<p>| J<sub>PAR</sub> |</p>')
        jPerpLbl = QtWidgets.QLabel('<p>| J<sub>PERP</sub> |</p>')

        # Set up quality measures layout
        qmFrame = QtWidgets.QFrame()
        qmFrame.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        qmLt = QtWidgets.QGridLayout(qmFrame)
        qmLbl = QtWidgets.QLabel('Quality Measures')

        qmLt.addWidget(divLbl, 0, 0, 1, 1)
        qmLt.addWidget(self.divB, 0, 1, 1, 1)
        qmLt.addWidget(ratioLbl, 1, 0, 1, 1)
        qmLt.addWidget(self.divBcurlB, 1, 1, 1, 1)

        # Insert all widgets into layout
        colNum = 0
        order = [(rLbl, 2), (curlLbl, 1), (jLbl, 2)]
        for e, spn in order:
            layout.addWidget(e, 0, colNum, 1, spn, alignment=QtCore.Qt.AlignCenter)
            colNum += spn

        colNum = 0
        order = [(rFrame, 2, 3), (curlFrame, 1, 3), (JFrame, 2, 3)]
        for e, spn, ht in order:
            layout.addWidget(e, 1, colNum, ht, spn)
            colNum += spn

        layout.addWidget(qmLbl, 4, 0, 1, 2, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(qmFrame, 5, 0, 2, 2)

        layout.addWidget(jParLbl, 4, 2, 1, 1, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(jPerpLbl, 4, 3, 1, 2, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(jParFrame, 5, 2, 2, 1)
        layout.addWidget(jPerpFrame, 5, 3, 2, 2)

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
        for i, e in enumerate([self.JxChk, self.JyChk, self.JzChk]):
            plotLt.addWidget(e)

        self.progressBar = QtWidgets.QProgressBar()
        self.applyBtn = QtWidgets.QPushButton('Apply')
        self.applyBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        layout.addWidget(plotFrame, 7, 0, 3, 3)
        layout.addWidget(self.applyBtn, 8, 3, 2, 1)
        layout.addWidget(self.progressBar, 8, 4, 2, 1)

        # Set up TimeEdit and checkbox to keep window on top of main win
        btmLt = QtWidgets.QHBoxLayout()
        self.timeEdit = TimeEdit(QtGui.QFont())
        minDt, maxDt = window.getMinAndMaxDateTime()
        self.timeEdit.setupMinMax((minDt, maxDt))
        self.onTopCheckBox = QtWidgets.QCheckBox('Stay On Top')
        btmLt.addWidget(self.timeEdit.start)
        btmLt.addWidget(self.onTopCheckBox)
        btmLt.addStretch()

        layout.addLayout(btmLt, 10, 0, 1, 3)

class Curlometer(QtGui.QFrame, CurlometerUI, MMSTools):
    def __init__(self, window, parent=None):
        super(Curlometer, self).__init__(parent)
        self.window = window

        MMSTools.__init__(self, window)
        self.ui = CurlometerUI()
        self.ui.setupUI(self, window)

        # Set up window-on-top settings
        if self.ui.onTopCheckBox.isChecked():
            self.toggleWindowOnTop(True)
        self.ui.onTopCheckBox.clicked.connect(self.toggleWindowOnTop)

        # Set up new plot variables interface
        self.ui.applyBtn.clicked.connect(self.addNewVars)
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
        refSpc = 4 # Current default for now
        mu0 = constants.mu_0

        # Determine data index corresponding to selected time
        tO, tE = self.window.getSelectionStartEndTimes()
        times = self.window.getTimes(self.window.DATASTRINGS[0], 0)[0]
        index = self.window.calcDataIndexByTime(times, tO)
        if index == len(times):
            index = index - 1

        if specIndex is not None:
            index = specIndex

        # Calculate R, I
        R = self.getRMatrix(refSpc, index)
        I = self.getIVec(refSpc, index)

        # Calculate curl(B) and J from R and I
        Rinv = np.linalg.inv(R)
        curlB = np.matmul(Rinv, I)
        J = (1/mu0) * curlB

        # Calculate quality measures
        divB = self.calcDivB(refSpc, index)
        divCurlB = abs(divB) / np.linalg.norm(curlB)

        # Calculate J_Parallel and J_Perpendicular magnitudes
        jPara, jPerp = self.calcJperpJpar(J, index)

        # Switch all results back to original units
        jPara = self.convertToOriginal(jPara)
        jPerp = self.convertToOriginal(jPerp)
        R = self.convertToOriginal(R, dtaType='pos')
        J = self.convertToOriginal(J)
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
            currB = self.convertToSI(self.getVec(scNum, index))
            B += currB
        B = B / 4
        B = B / np.linalg.norm(B)

        jPara = np.dot(J, B)
        jProj = (np.dot(J, B) / (np.dot(B,B))) * B # J proj onto B
        jPerp = np.linalg.norm(J - jProj)

        return jPara, jPerp

    def calcRange(self):
        # Get data indices to generate values for
        t0, t1 = self.window.minTime, self.window.maxTime
        times = self.window.getTimes(self.window.DATASTRINGS[0], 0)[0]
        i0 = self.window.calcDataIndexByTime(times, t0)
        i1 = self.window.calcDataIndexByTime(times, t1)
        numIndices = abs(i1 - i0)

        # Initialize progress bar settings/visibility
        progVal = 0
        progStep = 100 / numIndices
        self.setProgressVis(True)

        mat = np.zeros((6, numIndices))
        for i in range(i0, i1):
            # Calculate |J|, |J_perp|, |J_par| at every index and store it
            matIndex = i - i0
            J, jPar, jPerp = self.calculate(i)
            mat[:,matIndex] = np.array([np.linalg.norm(J), jPar, jPerp, J[0], J[1], J[2]])
            progVal += progStep
            self.ui.progressBar.setValue(progVal)

        # Update progress bar to show finished status
        QtCore.QTimer.singleShot(2000, self.setProgressVis)

        return mat

    def addNewVars(self):
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

        # Calculate |J|, |J_Par|, |J_Perp| and initialize the new varialbes
        resultMat = self.calcRange()
        for dstr in varsToAdd:
            index = list(self.nameDict.values()).index(dstr)
            self.window.initNewVar(dstr, resultMat[index], 'nA/m^2')

    def setProgressVis(self, b=False):
        # Sets progress bar / label as hidden or visible
        self.ui.progressBar.setVisible(b)