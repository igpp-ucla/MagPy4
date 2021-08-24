from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from .MagPy4UI import MatrixWidget, VectorWidget, TimeEdit
from .edit import Edit

from scipy import interpolate as scInter
import pyqtgraph as pg
import numpy as np
import functools
import bisect
import os

class SmoothingToolUI(object):
    def setupUI(self, Frame, window):
        self.Frame = Frame
        Frame.setWindowTitle('Smoothing Tool')
        Frame.resize(100, 100)
        layout = QtWidgets.QGridLayout(Frame)

        # Set up layout for choosing smoothing options
        settingsLt = QtWidgets.QHBoxLayout()
        self.smoothMethodBx = QtWidgets.QComboBox()
        self.smoothMethodBx.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        smoothOpts = ['Shift', 'Interpolate: Linear', 'Interpolate: Akima',
            'Interpolate: PCHIP']
        for opt in smoothOpts:
            self.smoothMethodBx.addItem(opt)
        settingsLt.addWidget(self.smoothMethodBx)
        settingsLt.addStretch()

        layout.addLayout(settingsLt, 0, 0, 1, 4)

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
        dstrsFrame.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        dstrsLt = QtWidgets.QHBoxLayout(dstrsFrame)
        for dstr in dstrs:
            chkbx = QtWidgets.QCheckBox(dstr)
            dstrsLt.addWidget(chkbx)
            self.dstrChkBoxes.append(chkbx)

        layout.addWidget(dstrsFrame, 1, 0, 1, 4)

        # Set up apply/undo buttons and 'keep on top' checkbox
        self.onTopCheckBox = QtWidgets.QCheckBox('Stay On Top')
        self.undoBtn = QtWidgets.QPushButton('Undo')
        self.applyBtn = QtWidgets.QPushButton('Apply')
        self.showLogBtn = QtWidgets.QPushButton('Show Log')

        layout.addWidget(self.onTopCheckBox, 2, 0, 1, 1)
        layout.addWidget(self.undoBtn, 2, 1, 1, 1)
        layout.addWidget(self.applyBtn, 2, 2, 1, 1)
        layout.addWidget(self.showLogBtn, 2, 3, 1, 1)

        # Save state info, connect show log button
        self.showLogBtn.clicked.connect(self.showLog)
        self.logWidget = None
        self.layout = layout

    def showLog(self):
        if self.logWidget is None:
            self.showLogBtn.setText('Close Log')
            # Create log and update
            self.logWidget = self.buildLogView()
            self.layout.addWidget(self.logWidget, 3, 0, 1, 4)
            self.Frame.updateLog()
            # Add in save log button
            self.saveBtn = QtWidgets.QPushButton('Save Log')
            self.saveBtn.clicked.connect(self.Frame.saveLog)
            self.layout.addWidget(self.saveBtn, 4, 3, 1, 1)
        else:
            # Close widgets and reset settings
            self.showLogBtn.setText('Show Log')
            self.logWidget.close()
            self.saveBtn.close()
            self.logWidget = None
            self.adjustWindowSize()

    def adjustWindowSize(self):
        # Adjusts window size after closing log widget
        QtCore.QTimer.singleShot(5, functools.partial(self.Frame.resize, 100, 100))

    def buildLogView(self):
        # Initializes the log widget
        textBox = QtWidgets.QTextEdit()
        textBox.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        textBox.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        textBox.setReadOnly(True)
        return textBox

    def updateLog(self, txt):
        if self.logWidget:
            self.logWidget.setText(txt)

class SmoothingTool(QtWidgets.QFrame, SmoothingToolUI):
    def __init__(self, window, editWindow, parent=None):
        super(SmoothingTool, self).__init__(parent)
        self.window = window
        self.lastEdits = [] # Stores per-edit lists of dstrs and num changes
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

        # Get list of last set of edits and update current state
        lastEdit = self.lastEdits[-1]
        for dstr, numChanges in lastEdit:
            # Remove last x elements from dstr's entry in change log
            for i in range(0, numChanges):
                self.window.changeLog[dstr].pop()
        self.lastEdits.pop()

        # Update actual log widget w/ new text
        self.updateLog()

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
        if len(self.window.currSelect.regions) == 0:
            return

        self.lastEdits.append([])

        # For every checked datastring, smooth its data
        for chkbx in self.ui.dstrChkBoxes:
            if not chkbx.isChecked():
                continue
            dstr = chkbx.text()
            en = self.window.currentEdit
            regTimes = []
            regions = []
            # Get region start/end indices
            for regNum in range(0, len(self.window.currSelect.regions)):
                iO, iE = self.window.calcDataIndicesFromLines(dstr, en, regNum)
                regions.append((iO, iE))
            
            dta = self.window.getData(dstr, en)
            times = self.window.getTimes(dstr, en)[0]

            smoothedDta = self.smoothDta(times, dta, regions)
            self.window.DATADICT[dstr].append(smoothedDta)

            # Record timestamps of change region
            regTimes = [(times[a], times[b]) for a, b in regions]
            if self.ui.smoothMethodBx.currentText() == 'Shift':
                # Use adjusted time region for shifts
                regTimes = []
                shiftRegions = [self.getRegion(dta, times, reg) for reg in regions]
                for a, b, c, d in shiftRegions:
                    regTimes.append((times[a], times[b+1]))
            self.recordChanges(regTimes, dstr)

        # Update num of edits, update plots, and restart general select
        self.editCount = self.editCount + 1
        self.window.endGeneralSelect()
        self.updatePlots()
        self.restartSelect()
        self.updateLog()

    def restartSelect(self):
        # Reset the general selection process and clear regions
        te = TimeEdit()
        self.window.closeTraceStats()
        self.window.initGeneralSelect('Smooth', '#4286f4', te, 'Single',
            None, maxSteps=-1)

    def updatePlots(self):
        self.edit.addHistory(np.eye(3), 'Data correction operation', 'S')

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
        # Interpolate rmRegion data and insert into rest of data
        rmStart, rmEnd = rmRegion
        rmDta, rmTimes, restOfDta, restOfTimes = self.splitDta(dta, times, rmStart, rmEnd)

        mode = self.ui.smoothMethodBx.currentText()[len('Interpolate: '):]

        # Interpolate according to specific method
        if mode == 'Linear':
            interpDta = np.interp(rmTimes, restOfTimes, restOfDta)
        elif mode == 'Akima':
            cs = scInter.Akima1DInterpolator(restOfTimes, restOfDta)
            interpDta = cs(rmTimes)
        elif mode == 'PCHIP':
            # Use subset of data for PCHIP for speed reasons
            minIndex = max(0, rmStart-200)
            maxIndex = min(len(restOfDta), rmStart+200)
            fitTimes = restOfTimes[minIndex:maxIndex]
            fitDta = restOfDta[minIndex:maxIndex]
            cs = scInter.PchipInterpolator(fitTimes, fitDta)
            interpDta = cs(rmTimes)

        newDta = np.insert(restOfDta, rmStart, interpDta)
        return newDta

    def smoothByShift(self, times, dta, rmRegion):
        rmStart, rmEnd, lftPeak, rghtPeak = self.getRegion(dta, times, rmRegion)
        rmDta, rmTimes, restOfDta, restOfTimes = self.splitDta(dta, times, rmStart, rmEnd)

        # Interpolate from rmStart to rmEnd to get lower interp line
        interpLower = np.interp(times[rmStart:rmEnd+1], restOfTimes, restOfDta)
        interpLower = interpLower[(lftPeak-rmStart):(rghtPeak-rmStart+1)]

        # Interpolate along the removed times between the value peaks
        shiftTimes = times[lftPeak:rghtPeak+1]
        shiftBoundaries = [times[lftPeak], times[rghtPeak]]
        shiftDtaBnds = [dta[lftPeak], dta[rghtPeak]]
        interpUpper = np.interp(shiftTimes, shiftBoundaries, shiftDtaBnds)

        # Calculate difference between these two
        diff = interpUpper - interpLower
        shiftedMiddle = dta[lftPeak:rghtPeak+1] - diff

        # Interpolate linearly in gaps between shifted section and original data
        interpLftTimes = times[rmStart:lftPeak]
        interpRghtTimes = times[rghtPeak+1:rmEnd+1]

        lftBnds, lftDta = [times[rmStart], times[lftPeak]], [dta[rmStart], shiftedMiddle[0]]
        rghtBnds, rghtDta = [times[rghtPeak+1], times[rmEnd]], [shiftedMiddle[-1], dta[rmEnd]]

        interpLeft = np.interp(interpLftTimes, lftBnds, lftDta)
        interpRght = np.interp(interpRghtTimes, rghtBnds, rghtDta)

        # Concatenate the interpolated gaps with the shifted middle section
        liftedDta = np.concatenate([interpLeft, shiftedMiddle, interpRght])

        # Insert full set of shifted data back into the larger data set
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
        return (lftChangePt-1, rghtChangePt+1, lftPeak, rightPeak)

    def getPeak(self, cp, slopes):
        peakIndex = cp + 1
        for i in range(peakIndex, len(slopes)-1):
            if np.sign(slopes[i]) != np.sign(slopes[i+1]):
                peakIndex = i + 1
                break
        return peakIndex

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

        # Find the inflection points and the nearest peak
        changePoints = []
        peakIndices = []
        for i in range(0, len(slopeDiffs)-1):
            if np.sign(slopeDiffs[i]) != np.sign(slopeDiffs[i+1]):
                changePoints.append(i+1)
                peakIndices.append(self.getPeak(i+1, slopes))

        if changePoints == []:
            return start, start

        # Find the inflection point with the greatest slope
        mostSignfPt = changePoints[0]
        peakIndex = peakIndices[0]
        for i in range(0, len(changePoints)):
            index = changePoints[i]
            currPeak = peakIndices[i]
            # Look at the slope between the inflection point and the next slope change
            diff = self.diffBetweenPts(dta, times, index, currPeak)
            if abs(diff) > abs(self.diffBetweenPts(dta, times, mostSignfPt, peakIndex)):
                mostSignfPt = index
                peakIndex = currPeak

        return start + mostSignfPt, start + peakIndex

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

        # Find the inflection points and the nearest peak
        changePoints = []
        peakIndices = []
        for i in range(0, len(slopeDiffs)-1):
            if np.sign(slopeDiffs[i]) != np.sign(slopeDiffs[i+1]):
                changePoints.append(i+1)
                peakIndices.append(self.getPeak(i+1, slopes))

        if changePoints == []:
            return end, end

        # Find the inflection point with greatest slope
        mostSignfPt = changePoints[0]
        peakIndex = peakIndices[0]
        for i in range(0, len(changePoints)):
            index = changePoints[i]
            currPeak = peakIndices[i]
            # Look at the slope between the inflection point and the next slope change
            diff = self.diffBetweenPts(dta, times, index, currPeak)
            if abs(diff) > abs(self.diffBetweenPts(dta, times, mostSignfPt, peakIndex)):
                mostSignfPt = index
                peakIndex = currPeak

        # Switch order for right edge b/c mostSignfPt is where slope changes
        return start + peakIndex, start + mostSignfPt

    def diffBetweenPts(self, dta, times, i1, i2):
        top = dta[i2] - dta[i1]
        bot = times[i2] - times[i1]
        return (top / bot)

    def recordChanges(self, timeTicks, dstr):
        # Save changes to change log and update internal edit history state
        if dstr in self.window.changeLog.keys():
            self.window.changeLog[dstr].extend(timeTicks)
        else:
            self.window.changeLog[dstr] = timeTicks
        # This edit's entry in lastEdits holds each dstr changed and the
        # number of regions that were changed
        self.lastEdits[-1].append((dstr, len(timeTicks)))

    def mergeOverlaps(self, indices):
        if len(indices) <= 1:
            return indices

        indices.sort() # Pre-sort indices

        # Check for time regions that overlap
        newIndexList = [indices[0]]
        prevA, prevB = newIndexList[0]
        for a, b in indices[1:]:
            # Merge time regions into a single time region if they overlap
            if prevA <= a and a <= prevB:
                mergedIndices = (prevA, max(b, prevB))
                newIndexList.pop()
            else:
                mergedIndices = (a, b)
            newIndexList.append(mergedIndices)
            prevA, prevB = mergedIndices

        return newIndexList

    def updateLog(self):
        txt = self.buildLogText()
        self.ui.updateLog(txt)

    def buildLogText(self):
        # Get list of changed data variables
        dstrs = list(self.window.changeLog.keys())
        if dstrs == []:
            return

        # Map time ticks for each dstr to timestamps
        timestamps = {}
        for dstr in dstrs:
            indices = self.window.changeLog[dstr]

            # Merge any overlapping timestamps and sort them
            indices = self.mergeOverlaps(indices)

            dstrTmstmps = []
            for a, b in indices:
                tmstmpA = self.window.getTimestampFromTick(a)
                tmstmpB = self.window.getTimestampFromTick(b)
                dstrTmstmps.append((tmstmpA, tmstmpB))

            timestamps[dstr] = dstrTmstmps

        # Create log text from timestamps dictionary
        txt = 'Log of Changes:\n'
        for dstr in dstrs:
            headerTxt = dstr + '\t' + 'Start, End' + '\n'

            bodyTxt = ''
            for tmA, tmB in timestamps[dstr]:
                bodyTxt += '\t' + tmA + ', ' + tmB + '\n'

            if bodyTxt == '':
                continue

            txt += headerTxt + bodyTxt # Move to new row

        return txt

    def saveLog(self):
        filename = self.window.saveFileDialog()
        f = open(filename, 'w')
        txt = self.buildLogText()
        f.write(txt)
        f.close()