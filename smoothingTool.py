from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from MagPy4UI import MatrixWidget, VectorWidget, TimeEdit, NumLabel
from dataDisplay import DataDisplay, UTCQDate
from edit import Edit
from FF_Time import FFTIME, leapFile

from scipy import interpolate as scInter

import pyqtgraph as pg
import numpy as np

import functools
import bisect

class SmoothingToolUI(object):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Smoothing Tool')
        Frame.resize(100, 100)
        layout = QtWidgets.QGridLayout(Frame)

        # Set up layout for choosing smoothing options
        settingsLt = QtWidgets.QHBoxLayout()
        self.smoothMethodBx = QtWidgets.QComboBox()
        self.smoothMethodBx.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        smoothOpts = ['Shift', 'Interpolate']
        for opt in smoothOpts:
            self.smoothMethodBx.addItem(opt)
        settingsLt.addWidget(self.smoothMethodBx)
        settingsLt.addStretch()

        layout.addLayout(settingsLt, 0, 0, 1, 3)

        # Get list of unique data strings in plots, in same order added
        allDstrs = []
        for lst in window.lastPlotStrings:
            for (dstr, en) in lst:
                allDstrs.append(dstr)
        dstrs = []
        for dstr in allDstrs:
            if dstr not in dstrs:
                dstrs.append(dstr)

        # Set up layout for listing which data vars will be affected
        self.dstrChkBoxes = []
        dstrsFrame = QtWidgets.QGroupBox('Select variables to smooth and press apply:')
        dstrsLt = QtWidgets.QHBoxLayout(dstrsFrame)
        for dstr in dstrs:
            chkbx = QtWidgets.QCheckBox(dstr)
            dstrsLt.addWidget(chkbx)
            self.dstrChkBoxes.append(chkbx)

        layout.addWidget(dstrsFrame, 1, 0, 1, 3)

        # Set up apply/undo buttons and 'keep on top' checkbox
        self.onTopCheckBox = QtWidgets.QCheckBox('Stay On Top')
        self.undoBtn = QtWidgets.QPushButton('Undo')
        self.applyBtn = QtWidgets.QPushButton('Apply')

        layout.addWidget(self.onTopCheckBox, 2, 0, 1, 1)
        layout.addWidget(self.applyBtn, 2, 2, 1, 1)
        layout.addWidget(self.undoBtn, 2, 1, 1, 1)

class SmoothingTool(QtGui.QFrame, SmoothingToolUI):
    def __init__(self, window, editWindow, parent=None):
        super(SmoothingTool, self).__init__(parent)
        self.window = window
        self.regions = []
        self.editCount = 0
        self.edit = editWindow

        self.ui = SmoothingToolUI()
        self.ui.setupUI(self, window)

        self.ui.applyBtn.clicked.connect(self.smooth)
        self.ui.undoBtn.clicked.connect(self.undoLastEdit)
        self.ui.onTopCheckBox.clicked.connect(self.toggleWindowOnTop)
        self.ui.onTopCheckBox.setChecked(True)
        self.toggleWindowOnTop(True)

    def closeEvent(self, event):
        self.window.endGeneralSelect()
        self.window.closeSmoothing()
        self.edit.close()
        self.close()

    def undoLastEdit(self):
        # Undo last added edit and replot data
        if self.editCount <= 0:
            return

        self.edit.removeHistory()
        self.editCount = self.editCount - 1

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

    def smooth(self):
        if len(self.window.regions) == 0:
            return

        # For every checked datastring, smooth its data
        for chkbx in self.ui.dstrChkBoxes:
            if not chkbx.isChecked():
                continue
            dstr = chkbx.text()
            en = self.window.currentEdit
            for regNum in range(0, len(self.window.regions)):
                iO, iE = self.window.calcDataIndicesFromLines(dstr, en, regNum)
                self.regions.append((iO, iE))
            
            dta = self.window.getData(dstr, en)
            times = self.window.getTimes(dstr, en)[0]

            smoothedDta = self.smoothDta(times, dta, self.regions)
            self.window.DATADICT[dstr].append(smoothedDta)

        # Update num of edits, update plots, and restart general select
        self.editCount = self.editCount + 1
        self.window.endGeneralSelect()
        self.updatePlots()
        self.restartSelect()

    def restartSelect(self):
        # Reset the general selection process and clear regions
        self.regions = []
        te = TimeEdit(QtGui.QFont())
        self.window.closeTraceStats()
        self.window.initGeneralSelect('Smooth', '#4286f4', te)

    def updatePlots(self):
        self.edit.addHistory(np.eye(3), 'Smoothing operation', 'S')
        self.editCount = self.editCount + 1

    def smoothDta(self, times, dta, rmRegions):
        # Smooth data in every region, updating data used w/ each iteration
        newDta = dta
        mode = self.ui.smoothMethodBx.currentText()
        for rmRegion in rmRegions:
            if mode == 'Shift':
                newDta = self.smoothByShift(times, newDta, rmRegion)
            else:
                newDta = self.smoothByInterpolation(times, newDta, rmRegion)
        return newDta

    def smoothByInterpolation(self, times, dta, rmRegion):
        # Linearly interpolate rmRegion data and insert into rest of data
        rmStart, rmEnd = rmRegion
        rmDta, rmTimes, restOfDta, restOfTimes = self.splitDta(dta, times, rmStart, rmEnd)
        interpDta = np.interp(rmTimes, restOfTimes, restOfDta)
        newDta = np.insert(restOfDta, rmStart, interpDta)
        return newDta

    def smoothByShift(self, times, dta, rmRegion):
        rmStart, rmEnd, lftPeak, rghtPeak = self.getRegion(dta, times, rmRegion)
        rmDta, rmTimes, restOfDta, restOfTimes = self.splitDta(dta, times, rmStart, rmEnd)

        interpDta = np.interp(rmTimes, restOfTimes, restOfDta)

        shiftTimes = times[lftPeak:rghtPeak+1]
        shiftBoundaries = [times[lftPeak], times[rghtPeak]]
        shiftDtaBnds = [dta[lftPeak], dta[rghtPeak]]

        # Interpolate along the removed times between the value peaks
        interpInner = np.interp(shiftTimes, shiftBoundaries, shiftDtaBnds)

        # Add back in some points for purposes of finding difference
        numLftPts = lftPeak-rmStart
        numRghtPts = rmEnd-rghtPeak
        interpLeft = interpDta[:numLftPts]
        interpRight = interpDta[-numRghtPts:]
        interpInner = np.concatenate([interpLeft, interpInner, interpRight])

        # Find the difference between the top interp line and the bottom one
        diff = interpInner - interpDta

        # Replace the gaps between the rm bounds and peak bounds w/ bottom dta
        liftedDta = rmDta - diff
        liftedDta[:len(interpLeft)] = interpLeft
        liftedDta[-len(interpRight):] = interpRight

        # Insert shifted data back into the larger data set
        newDta = np.insert(restOfDta, rmStart, liftedDta)
        return newDta

    def splitDta(self, dta, times, rmStart, rmEnd):
        # Separate data into rmTimes, rmDta, nonRmDta, nonRmTimes
        rmIndices = [i for i in range(rmStart, rmEnd+1)]
        rmDta = dta[rmIndices]
        rmTimes = [times[i] for i in rmIndices]
        restOfDta = np.delete(dta, rmIndices)
        restOfTimes = np.delete(times, rmIndices)
        return rmDta, rmTimes, restOfDta, restOfTimes

    def getRegion(self, dta, times, selectedRegion):
        lowerIndex, upperIndex = selectedRegion

        # Get upper quartile and lower quartile of selected region
        numPoints = upperIndex - lowerIndex
        qtrRange = max(2, int(numPoints/4))
        halfRange = max(2, int(numPoints/2))
        middlePt = lowerIndex + halfRange

        lftRegion = (lowerIndex, middlePt)
        rghtRegion = (middlePt, upperIndex)

        # Calculate the points where the slope changes and where the value
        # peaks just after/before this point
        lftChangePt, lftPeak = self.getLeftChangePoint(dta, times, lftRegion)
        rghtChangePt, rightPeak = self.getRightChangePoint(dta, times, rghtRegion)

        # Return slightly extended regions
        return (lftChangePt - 2, rghtChangePt + 2, lftPeak, rightPeak)

    def getLeftChangePoint(self, dta, times, region):
        # Look at region dta/times only
        start, end = region
        dta = dta[start:end+1]
        times = times[start:end+1]

        # Scale times for precision purposes
        times = times * 100

        # Get slopes and their differences
        slopes = np.diff(dta) / np.diff(times)
        slopeDiffs = np.diff(slopes)

        # Find the inflection points
        changePoints = []
        for i in range(0, len(slopeDiffs)-1):
            if np.sign(slopeDiffs[i]) != np.sign(slopeDiffs[i+1]):
                changePoints.append(i+1)

        if changePoints == []:
            return start, start

        # Find the inflection point with the greatest slope
        mostSignfPt = changePoints[0]
        for index in changePoints:
            diff = slopes[index]
            if abs(diff) > abs(slopes[mostSignfPt]):
                mostSignfPt = index

        # Find peak, first point where slope sign changes after inflection point
        peakIndex = mostSignfPt
        for i in range(mostSignfPt, len(slopes)-1):
            if np.sign(slopes[i]) != np.sign(slopes[i+1]):
                peakIndex = i + 1
                break

        return start + mostSignfPt - 1, start + peakIndex

    def getRightChangePoint(self, dta, times, region):
        # Look at region dta/times only
        start, end = region
        dta = dta[start:end+1]
        times = times[start:end+1]

        # Scale times for precision purposes
        times = times * 100

        # Get slopes and their differences
        slopes = np.diff(dta) / np.diff(times)
        slopeDiffs = np.diff(slopes)

        # Find the inflection points
        changePoints = []
        for i in range(0, len(slopeDiffs)-1):
            if np.sign(slopeDiffs[i]) != np.sign(slopeDiffs[i+1]):
                changePoints.append(i+1)

        if changePoints == []:
            return end, end

        # Find the inflection point with greatest slope
        mostSignfPt = changePoints[-1]
        for index in changePoints:
            diff = slopes[index]
            if abs(diff) > abs(slopes[mostSignfPt]):
                mostSignfPt = index

        # Find peak, first point where slope sign changes before inflection point
        peakIndex = mostSignfPt - 1
        for i in range(mostSignfPt - 1, 0, -1):
            if np.sign(slopes[i]) != np.sign(slopes[i+1]):
                peakIndex = i + 1
                break

        return start + mostSignfPt + 1, start + peakIndex


