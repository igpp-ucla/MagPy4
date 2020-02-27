from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from .plotAppearance import DynamicPlotApp
import pyqtgraph as pg
from scipy import fftpack, signal
import numpy as np
from .MagPy4UI import TimeEdit, NumLabel
from .pyqtgraphExtensions import GridGraphicsLayout, LogAxis, MagPyAxisItem, DateAxis, StackedAxisLabel
import bisect
import functools
from .mth import Mth
from .layoutTools import BaseLayout
from .simpleCalculations import ExpressionEvaluator
from .dynBase import DynamicAnalysisTool, SpectraLineEditor, SpectrogramPlotItem, SpectraLegend, SpectraBase, PhaseSpectrogram
import os

class DynamicSpectraUI(BaseLayout):
    def setupUI(self, Frame, window):
        self.Frame = Frame
        maxSizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        Frame.setWindowTitle('Dynamic Spectrogram')
        Frame.resize(1050, 850)
        layout = QtWidgets.QGridLayout(Frame)

        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout(window)
        self.gview.setCentralItem(self.glw)
        self.glw.layout.setHorizontalSpacing(5)

        # Settings setup
        settingsFrame = QtWidgets.QGroupBox(' Settings')
        settingsLt = QtWidgets.QGridLayout(settingsFrame)

        lbls = ['FFT Interval: ', 'FFT Shift: ', 'Bandwidth: ', '   Data Variable: ',
            '   Scaling Mode: ', '   Data Points: ']

        self.fftDataPts = QtWidgets.QLabel()
        self.fftInt = QtWidgets.QSpinBox()
        self.fftShift = QtWidgets.QSpinBox()
        self.bwBox = QtWidgets.QSpinBox()
        self.bwBox.setSingleStep(2)
        self.bwBox.setMinimum(1)
        self.dstrBox = QtWidgets.QComboBox()
        self.scaleModeBox = QtWidgets.QComboBox()

        dtaPtsTip = 'Total number of data points within selected time range'
        shiftTip = 'Number of data points to move forward after each FFT calculation'
        fftIntTip = 'Number of data points to use per FFT calculation'
        scaleTip = 'Scaling mode that will be used for y-axis (frequencies)'
        detrendTip = 'Detrend data using the least-squares fit method before each FFT is applied '

        spacer = QtWidgets.QSpacerItem(10, 1)

        self.updateBtn = QtWidgets.QPushButton('Update')
        self.updateBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.addLineBtn = QtWidgets.QPushButton('Add Line')
        self.addLineBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.maskBtn = QtWidgets.QPushButton('Mask')
        self.maskBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        self.detrendCheck = QtWidgets.QCheckBox(' Detrend Data')
        self.detrendCheck.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.detrendCheck.setToolTip(detrendTip)

        # Set up FFT parameter widgets
        self.addPair(settingsLt, lbls[0], self.fftInt, 0, 0, 1, 1, fftIntTip)
        self.addPair(settingsLt, lbls[1], self.fftShift, 1, 0, 1, 1, shiftTip)
        self.addPair(settingsLt, lbls[2], self.bwBox, 2, 0, 1, 1)

        # Set up data variable parameters
        self.addPair(settingsLt, lbls[3], self.dstrBox, 0, 2, 1, 1)
        self.addPair(settingsLt, lbls[4], self.scaleModeBox, 1, 2, 1, 1, scaleTip)

        # Set up data points / detrend checkbox layout
        ptsLt = QtWidgets.QGridLayout()
        ptsLt.addWidget(self.fftDataPts, 0, 0, 1, 1)
        spacer = self.getSpacer(10)
        ptsLt.addItem(spacer, 0, 1, 1, 1)
        ptsLt.addWidget(self.detrendCheck, 0, 2, 1, 1)

        # Add in data points label and its sublayout to settings layout
        ptsLbl = QtWidgets.QLabel(lbls[5])
        ptsLbl.setToolTip(dtaPtsTip)
        settingsLt.addWidget(ptsLbl, 2, 2, 1, 1)
        settingsLt.addLayout(ptsLt, 2, 3, 1, 1)

        spacer = QtWidgets.QSpacerItem(10, 1)
        settingsLt.addItem(spacer, 3, 4, 1, 1)

        # Set up power min/max checkbox layout
        powerRngLt = self.setPowerRangeLt()
        self.powerRngSelectToggled(False)
        settingsLt.addLayout(powerRngLt, 0, 5, 3, 1)

        spacer = QtWidgets.QSpacerItem(10, 1)
        settingsLt.addItem(spacer, 2, 6, 1, 1)
        settingsLt.addWidget(self.updateBtn, 2, 7, 1, 1)
        settingsLt.addWidget(self.addLineBtn, 1, 7, 1, 1)
        settingsLt.addWidget(self.maskBtn, 0, 7, 1, 1)

        layout.addWidget(settingsFrame, 0, 0, 1, 2)

        # Set up time edits at bottom of window + status bar
        timeLt, self.timeEdit, self.statusBar = self.getTimeStatusBar()
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())

        layout.addLayout(timeLt, 9, 0, 1, 2)

        # Initialize data values for spinboxes and combo boxes
        itemSet = set()
        for plt in window.lastPlotStrings:
            for (dstr, en) in plt:
                if dstr not in itemSet and en >= 0: # Skip duplicates
                    self.dstrBox.addItem(dstr)
                itemSet.add(dstr)
        self.dstrBox.insertSeparator(len(itemSet))
        if len(itemSet) >= 3:
            self.dstrBox.addItems(Frame.sumPowersPlotTypes)

        # Set up plot scaling combo box options
        self.scaleModeBox.addItem('Logarithmic')
        self.scaleModeBox.addItem('Linear')

        # Plot setup
        layout.addWidget(self.gview, 1, 0, 8, 2)

        # Default settings
        self.bwBox.setValue(3)

    def initVars(self, window):
        # Initializes the max/min times
        minTime, maxTime = window.getTimeTicksFromTimeEdit(self.timeEdit)
        dstr = self.dstrBox.currentText()
        if dstr in self.Frame.spStateKws:
            dstr = self.Frame.getVecDstrs(dstr)[0]
        times = window.getTimes(dstr, 0)[0]
        startIndex = window.calcDataIndexByTime(times, minTime)
        endIndex = window.calcDataIndexByTime(times, maxTime)
        numPoints = abs(endIndex-startIndex)
        self.fftDataPts.setText(str(numPoints))

    def setPowerRangeLt(self):
        self.selectToggle = QtWidgets.QCheckBox(' Set Power Range: ')
        rangeLt = QtWidgets.QGridLayout()

        rngTip = 'Toggle to set max/min power levels represented by color gradient'
        self.selectToggle.setToolTip(rngTip)

        minTip = 'Minimum power level represented by color gradient'
        maxTip = 'Maximum power level represented by color gradient'

        self.valueMin = QtWidgets.QDoubleSpinBox()
        self.valueMax = QtWidgets.QDoubleSpinBox()

        # Set spinbox defaults
        for box in [self.valueMax, self.valueMin]:
            box.setPrefix('10^')
            box.setMinimum(-100)
            box.setMaximum(100)
            box.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        spc = '       ' # Spaces that keep spinbox lbls aligned w/ chkbx lbl

        rangeLt.addWidget(self.selectToggle, 0, 0, 1, 2)
        self.maxLbl = self.addPair(rangeLt, spc+'Max: ', self.valueMax, 1, 0, 1, 1, maxTip)
        self.minLbl = self.addPair(rangeLt, spc+'Min: ', self.valueMin, 2, 0, 1, 1, minTip)

        # Connects checkbox to func that enables/disables rangeLt's items
        self.selectToggle.toggled.connect(self.powerRngSelectToggled)
        return rangeLt

    def powerRngSelectToggled(self, val):
        self.valueMax.setEnabled(val)
        self.valueMin.setEnabled(val)
        self.minLbl.setEnabled(val)
        self.maxLbl.setEnabled(val)

class DynamicSpectra(QtGui.QFrame, DynamicSpectraUI, DynamicAnalysisTool):
    def __init__(self, window, parent=None):
        super(DynamicSpectra, self).__init__(parent)
        SpectraBase.__init__(self)
        DynamicAnalysisTool.__init__(self)
        self.plotItem = None

        # Set up sum of powers plot titles and simple keywords that they map
        # to elsewhere in the DynamicSpectra class
        self.sumPowersPlotTypes = ['|Px + Py + Pz - Pt|', 'Px + Py + Pz', 'Pt']
        stateKws = ['AbsSumPowers', 'SumPowers', 'MagPower']
        self.spStateKws = {}
        for kw, stateKw in zip(self.sumPowersPlotTypes, stateKws):
            self.spStateKws[kw] = stateKw
        # Find any plotted vector groups beforehand
        self.vecGrps = window.findPlottedVecGroups()
        self.vecIdentifiers = self.getVecIdentifiers(self.vecGrps)
        # If multiple vectors are plotted, add identifiers to the sum of 
        # powers plot titles/strings and add them to the state dictionary
        if len(self.vecIdentifiers) != 0:
            newPlotTypes = []
            for substr in self.vecIdentifiers:
                for kw, stateKw in zip(self.sumPowersPlotTypes, stateKws):
                    newKw = kw + ' (' + substr + ')'
                    self.spStateKws[newKw] = stateKw
                    newPlotTypes.append(newKw)
            self.sumPowersPlotTypes = newPlotTypes

        self.ui = DynamicSpectraUI()
        self.window = window
        self.wasClosed = False

        self.gradRange = None # Custom color gradient range
        self.lastCalc = None # Previously calculated values, if any

        self.ui.setupUI(self, window)
        self.ui.updateBtn.clicked.connect(self.update)
        self.ui.addLineBtn.clicked.connect(self.openLineTool)
        self.ui.maskBtn.clicked.connect(self.openMaskTool)

    def closeEvent(self, ev):
        self.closeLineTool()
        self.closeMaskTool()
        self.window.endGeneralSelect()
        if self.plotItem:
            self.plotItem.closePlotAppearance()
        self.window.clearStatusMsg()
        self.wasClosed = True

    def getToolType(self):
        return 'Spectra'

    def getDataRange(self):
        dstr = self.ui.dstrBox.currentText()
        if dstr in self.spStateKws:
            dstr = self.getVecDstrs(dstr)[0]

        # Get selection times and convert to corresponding data indices for dstr
        minTime, maxTime = self.window.getSelectionStartEndTimes()
        times = self.window.getTimes(dstr, 0)[0]
        startIndex = self.window.calcDataIndexByTime(times, minTime)
        endIndex = self.window.calcDataIndexByTime(times, maxTime)

        return startIndex, endIndex

    def getVarInfo(self):
        return self.ui.dstrBox.currentText()

    def setVarParams(self, var):
        self.ui.dstrBox.setCurrentText(var)

    def update(self):
        # Determine data indices from lines
        dataRng = self.getDataRange()
        numPoints = dataRng[1] - dataRng[0]

        # Extract user-set parameters
        interval = self.ui.fftInt.value()
        shift = self.ui.fftShift.value()
        dstr = self.ui.dstrBox.currentText()
        bw = self.ui.bwBox.value()
        detrendMode = self.ui.detrendCheck.isChecked()
        logScaling = self.ui.scaleModeBox.currentText() == 'Logarithmic'
        colorRng = self.getGradRange()
        fftParam = (interval, shift, bw)

        # Error checking for user parameters
        if self.checkParameters(interval, shift, bw, numPoints) == False:
            return

        # Calculate grid values and set up the plot layout
        grid, freqs, times = self.calcGrid(dataRng, fftParam, dstr, detrendMode)
        if colorRng is None: # Default range is the range of vals in the grid
            minPower = np.min(grid[grid>0])
            maxPower = np.max(grid[grid>0])
            colorRng = (minPower, maxPower)
        plt = self.generatePlot(grid, freqs, times, colorRng, logScaling)
        self.setupPlotLayout(plt, dstr, times, logScaling)

        # Store calculations for displaying values at a point
        self.lastCalc = (times, freqs, grid)

        # Update min / max color map boxes
        self.updateMinMaxBoxes(grid)

        # Enable context menu option for saving plot data
        self.enableDataExport(interval, shift, bw, detrendMode)

        if self.savedLineInfo:
            self.addSavedLine()
        elif len(self.lineInfoHist) > 0 and len(self.lineHistory) == 0:
            self.savedLineInfo = self.lineInfoHist
            self.lineInfoHist = []
            self.addSavedLine()
            self.savedLineInfo = None

    def getGradRange(self):
        if self.ui.selectToggle.isChecked():
            minGradPower = self.ui.valueMin.value()
            maxGradPower = self.ui.valueMax.value()
            # Adjust order if flipped
            minGradPower = min(minGradPower, maxGradPower)
            maxGradPower = max(minGradPower, maxGradPower)
            gradRange = (10**minGradPower, 10**maxGradPower)
        else:
            gradRange = None
        return gradRange

    def updateMinMaxBoxes(self, grid):
        # Update min and max values in 'Set Range' boxes with the grid min/max
        # vals if the user did not specify a range
        minPower = np.min(grid[grid>0])
        maxPower = np.max(grid[grid>0])
        if not self.ui.selectToggle.isChecked():
            self.ui.valueMin.setValue(np.log10(minPower))
            self.ui.valueMax.setValue(np.log10(maxPower))

    def enableDataExport(self, interval, shift, bw, detrend):
        # Enable exporting plot data
        fftParam = (interval, shift, bw, detrend)
        exportFunc = functools.partial(self.exportData, self.window, 
            self.plotItem, fftParam)
        self.plotItem.setExportEnabled(exportFunc)

    def getVecIdentifiers(self, vecGrps):
        if len(vecGrps) == 1:
            return []

        # Attempts to get suffix corresponding to each unique vector
        substrs = []
        for grp in vecGrps:
            firstDstr = grp[0]
            substr = firstDstr.strip('Bx').strip('bx').strip('BX')
            if substr[0] == '_' or substr[0] == ' ':
                substr = substr[1:]
            substrs.append(substr)
        return substrs

    def getVecDstrs(self, dstr):
        if len(self.vecGrps) == 0: # Nonstandard file, use first 3 dstrs
            return [self.ui.dstrBox.itemText(i) for i in range(0, 3)]
        # Use index corresponding to vec's identifier or the first vec
        # if only one vector is plotted
        if len(self.vecIdentifiers) != 0:
            index = 0
            while index < len(self.vecIdentifiers):
                if self.vecIdentifiers[index] in dstr:
                    break
                index += 1
        else:
            index = 0
        return self.vecGrps[index]

    def calcSumOfPowers(self, vecDstrs, bw, startIndex, endIndex, detrend=False):
        # Calculates the spectra for each variable separately and returns the sum
        powers = []
        for dstr in vecDstrs:
            freqs, power = self.calcSpectra(dstr, bw, startIndex, endIndex, detrend=detrend)
            powers.append(np.array(power))

        sumOfPowers = powers[0] + powers[1] + powers[2]
        return freqs, sumOfPowers

    def calcMag(self, vecDstrs, startIndex, endIndex, detrend=False):
        a, b = startIndex, endIndex
        # Computes the magnitude of the vector over the selected interval
        en = self.window.currentEdit
        datas = [self.window.getData(dstr, en)[a:b] ** 2 for dstr in vecDstrs]
        mag = np.sqrt(datas[0] + datas[1] + datas[2])
        if detrend:
            mag = signal.detrend(mag)
        return mag

    def calcMagPower(self, vecDstrs, bw, start, end, detrend=False):
        # Similar to calcSpectra but using pre-computed magnitude data
        en = self.window.currentEdit
        N = abs(end-start)
        data = self.calcMag(vecDstrs, start, end, detrend=detrend)
        fft = fftpack.rfft(data.tolist())
        power = self.calculatePower(bw, fft, N)
        freqs = self.calculateFreqList(bw, N)
        return freqs, power

    def calcGrid(self, dataRng, fftParam, dstr, detrend=False):
        interval, shift, bw = fftParam
        shiftAmnt = shift
        minIndex, maxIndex = dataRng
        startIndex, endIndex = minIndex, minIndex + interval

        # Check if this is a special sum-of-powers plot
        spPlot = self.spStateKws[dstr] if dstr in self.spStateKws else None

        # Get vector var names and computer magnitude of vector if necessary
        if spPlot is not None:
            vecDstrs = self.getVecDstrs(dstr)
            dstr = vecDstrs[0] # Adjusted for future call to getTimes()

        # Generate the list of powers and times associated w/ each FFT interval
        powerLst = []
        timeSeries = []
        times = self.window.getTimes(dstr, 0)[0]
        while endIndex < maxIndex:
            # Save start time
            timeSeries.append(times[startIndex])
            # Calculate freqs and powers
            if spPlot is not None:
                if spPlot == 'SumPowers':
                    freqs, powers = self.calcSumOfPowers(vecDstrs, bw, startIndex, 
                        endIndex, detrend=detrend)
                elif spPlot == 'MagPower':
                    freqs, powers = self.calcMagPower(vecDstrs, bw, startIndex,
                        endIndex, detrend=detrend)
                else:
                    freqs, sumPowers = self.calcSumOfPowers(vecDstrs, bw, startIndex, 
                        endIndex, detrend=detrend)
                    freqs, magPower = self.calcMagPower(vecDstrs, bw, startIndex, 
                        endIndex, detrend=detrend)
                    powers = np.abs(sumPowers - magPower)
            else:
                freqs, powers = self.calcSpectra(dstr, bw, startIndex, endIndex, detrend)
            powerLst.append(np.array(powers))
            # Move to next interval
            startIndex += shiftAmnt
            endIndex = startIndex + interval
        timeSeries.append(times[startIndex])
        timeSeries = np.array(timeSeries)

        # Transpose here to turn fft result rows into per-time columns
        pixelGrid = np.array(powerLst)
        pixelGrid = pixelGrid.T

        return pixelGrid, freqs, timeSeries

    def generatePlot(self, grid, freqs, times, colorRng, logScaling):
        freqs = self.extendFreqs(freqs, logScaling) # Get lower bounding frequency
        plt = SpectrogramPlotItem(self.window.epoch, logScaling)
        plt.createPlot(freqs, grid, times, colorRng, winFrame=self)
        self.plotItem = plt
        return plt

    def setupPlotLayout(self, plt, dstr, times, logScaling):
        # Create gradient legend and add it to the graphics layout
        gradLegend = plt.getGradLegend()
        gradLegend.setBarWidth(40)

        # Get labels
        title, axisLbl, legendLbl = self.getLabels(dstr, logScaling)
        plt.setTitle(title, size='14pt')
        plt.getAxis('left').setLabel(axisLbl)

        # Time range information
        timeInfo = self.getTimeInfoLbl((times[0], times[-1]))

        self.ui.glw.clear()
        self.ui.glw.addItem(plt, 0, 0, 1, 1)
        self.ui.glw.addItem(gradLegend, 0, 1, 1, 1)
        self.ui.glw.addItem(legendLbl, 0, 2, 1, 1)
        self.ui.glw.addItem(timeInfo, 1, 0, 1, 3)

    def calcSpectra(self, dstr, bw, start, end, detrend=False):
        """
        Calculate the spectra for the given sub interval
        """
        # Convert time range to indices
        i0, i1 = start, end
        en = self.window.currentEdit
        N = abs(i1-i0)

        fft = self.getfft(dstr, en, i0, i1, detrend=detrend)
        power = self.calculatePower(bw, fft, N)
        freqs = self.calculateFreqList(bw, N)
        return freqs, power
    
    def extendFreqs(self, freqs, logScale):
        # Calculate frequency that serves as lower bound for plot grid
        diff = abs(freqs[1] - freqs[0])
        lowerFreqBnd = freqs[0] - diff
        if lowerFreqBnd == 0 and logScale:
            lowerFreqBnd = freqs[0] - diff/2
        freqs = np.concatenate([[lowerFreqBnd], freqs])
        return freqs

    def getLabels(self, dstr, logScale):
        title = 'Dynamic Spectra Analysis ' + '(' + dstr + ')'
        axisLbl = 'Frequency (Hz)'
        if logScale:
            axisLbl = 'Log ' + axisLbl
        legendLbl = StackedAxisLabel(['Log Power', '(nT^2/Hz)'], angle=0)
        legendLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))
        return title, axisLbl, legendLbl

    def showPointValue(self, freq, time):
        self.showValue(freq, time, 'Freq, Power: ', self.lastCalc)

    def addLineToPlot(self, line):
        self.plotItem.addItem(line)
        self.lineHistory.add(line)

    def removeLinesFromPlot(self):
        histCopy = self.lineHistory.copy()
        for line in histCopy:
            if line in self.plotItem.listDataItems():
                self.plotItem.removeItem(line)
                self.lineHistory.remove(line)

class DynamicCohPhaUI(BaseLayout):
    def setupUI(self, Frame, window, dstrs):
        maxSizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        Frame.setWindowTitle('Dynamic Coherence/Phase Analysis')
        Frame.resize(900, 800)
        layout = QtWidgets.QGridLayout(Frame)

        # Set up calculation settings layout
        settingsLt = self.setupSettingsLt(dstrs)
        layout.addLayout(settingsLt, 0, 0, 1, 2)

        # Set up tabs and grids for coh/pha plots
        self.tabWidget, self.cohGrid, self.phaGrid = self.setupTabs(window)
        layout.addWidget(self.tabWidget, 1, 0, 1, 2)

        # Set up status bar and time edits
        timeLt, self.timeEdit, self.statusBar = self.getTimeStatusBar()
        minDt, maxDt = window.getMinAndMaxDateTime()
        self.timeEdit.setupMinMax((minDt, maxDt))

        layout.addLayout(timeLt, 2, 0, 1, 2)

    def initVars(self, window):
        # Set num data points
        minTime, maxTime = window.getTimeTicksFromTimeEdit(self.timeEdit)
        times = window.getTimes(self.boxA.currentText(), 0)[0]
        startIndex = window.calcDataIndexByTime(times, minTime)
        endIndex = window.calcDataIndexByTime(times, maxTime)
        nPoints = abs(endIndex-startIndex)
        self.fftDataPts.setText(str(nPoints))

    def setupTabs(self, window):
        tabWidget = QtWidgets.QTabWidget()

        grids = []
        for name in ['Coherence', 'Phase']:
            subFrame = QtWidgets.QFrame()
            subLt = QtWidgets.QGridLayout(subFrame)

            # Build grid graphics layout
            gview = pg.GraphicsView()
            gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
            glw = GridGraphicsLayout(window)
            gview.setCentralItem(glw)
            grids.append(glw)
            subLt.addWidget(gview)
            glw.layout.setHorizontalSpacing(10)

            # Create a tab
            tabWidget.addTab(subFrame, name)

        return tabWidget, grids[0], grids[1]

    def setupSettingsLt(self, dstrs):
        layout = QtWidgets.QGridLayout()
        self.fftInt = QtWidgets.QSpinBox()
        self.fftShift = QtWidgets.QSpinBox()
        self.bwBox = QtWidgets.QSpinBox()

        # Set up FFT parameters layout
        lbls = ['FFT Interval: ', 'FFT Shift: ', 'Bandwidth: ']

        fftIntTip = 'Number of data points to use per FFT calculation'
        shiftTip = 'Number of data points to move forward after each FFT calculation'
        scaleTip = 'Scaling mode that will be used for y-axis (frequencies)'
        detrendTip = 'Detrend data using the least-squares fit method before each FFT is applied '
        ptsTip = 'Total number of data points within selected time range'

        tips = [fftIntTip, shiftTip, '']
        boxes = [self.fftInt, self.fftShift, self.bwBox]
        for i in range(0, 3):
            self.addPair(layout, lbls[i], boxes[i], i, 0, 1, 1, tips[i])
        layout.addItem(self.getSpacer(10), 0, 2, 3, 1)

        # Set up frequency scaling mode box
        self.scaleModeBox = QtWidgets.QComboBox()
        self.scaleModeBox.addItem('Logarithmic')
        self.scaleModeBox.addItem('Linear')
        self.addPair(layout, 'Scaling mode: ', self.scaleModeBox, 1, 3, 1, 1, scaleTip)

        # Set up pair chooser
        varLt = QtWidgets.QHBoxLayout()
        self.boxA = QtWidgets.QComboBox()
        self.boxB = QtWidgets.QComboBox()
        varLbl = QtWidgets.QLabel('Variable Pair: ')
        varLt.addWidget(self.boxA)
        varLt.addWidget(self.boxB)
        layout.addWidget(varLbl, 0, 3, 1, 1)
        layout.addLayout(varLt, 0, 4, 1, 1)

        for dstr in dstrs:
            self.boxA.addItem(dstr)
            self.boxB.addItem(dstr)
        if len(dstrs) > 1:
            self.boxB.setCurrentIndex(1)

        # Set up num points label, bandwidth btn, and update btn
        # Also, add in detrend check box within data pts layout
        self.detrendCheck = QtWidgets.QCheckBox(' Detrend Data')
        self.detrendCheck.setToolTip(detrendTip)
        self.fftDataPts = QtWidgets.QLabel('')
        ptsLbl = QtWidgets.QLabel('Data Points: ')
        ptsLbl.setToolTip(ptsTip)

        ptsLt = QtWidgets.QGridLayout()
        ptsLt.addWidget(self.fftDataPts, 0, 0, 1, 1)
        ptsLt.addWidget(self.detrendCheck, 0, 1, 1, 1)
        layout.addWidget(ptsLbl, 2, 3, 1, 1)
        layout.addLayout(ptsLt, 2, 4, 1, 1)

        self.bwBox.setValue(3)
        self.bwBox.setSingleStep(2)
        self.bwBox.setMinimum(1)

        self.addLineBtn = QtWidgets.QPushButton('Add Line')
        self.addLineBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.updtBtn = QtWidgets.QPushButton(' Update ')
        self.updtBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.maskBtn = QtWidgets.QPushButton('Mask')
        self.maskBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Column of spacers before update button
        for row in range(0, 3):
            spacer = self.getSpacer(10)
            layout.addItem(spacer, row, 6, 1, 1)

        layout.addWidget(self.updtBtn, 2, 7, 1, 1)
        layout.addWidget(self.addLineBtn, 1, 7, 1, 1)
        layout.addWidget(self.maskBtn, 0, 7, 1, 1)
        return layout

class DynamicCohPha(QtGui.QFrame, DynamicCohPhaUI, DynamicAnalysisTool):
    def __init__(self, window, parent=None):
        super(DynamicCohPha, self).__init__(parent)
        SpectraBase.__init__(self)
        DynamicAnalysisTool.__init__(self)
        self.ui = DynamicCohPhaUI()
        self.window = window
        self.wasClosed = False

        self.cohPlt, self.phaPlt = None, None
        self.gradRange = None # Custom color gradient range
        self.lastCalc = None # Previously calculated values, if any

        # Get full set of dstrs currently plotted and sort them
        dstrs = []
        for pltLst in self.window.lastPlotStrings:
            for dstr, en in pltLst:
                if dstr not in dstrs and en >= 0:
                    dstrs.append(dstr)

        # Close if nothing is currently being displayed
        if len(dstrs) == 0 or dstrs[0] == '':
            self.window.showStatusMsg('Error: Need at least one variable plotted.')
            self.close()

        self.ui.setupUI(self, window, dstrs)
        self.ui.updtBtn.clicked.connect(self.update)
        self.ui.addLineBtn.clicked.connect(self.openLineTool)
        self.ui.maskBtn.clicked.connect(self.openMaskTool)

    def closeEvent(self, ev):
        self.closeLineTool()
        self.closeMaskTool()
        self.window.endGeneralSelect()
        if self.cohPlt:
            self.cohPlt.closePlotAppearance()
        if self.phaPlt:
            self.phaPlt.closePlotAppearance()
        self.window.clearStatusMsg()
        self.wasClosed = True
        self.close()

    def getToolType(self):
        if self.ui.tabWidget.currentIndex() == 0:
            return 'Coherence'
        else:
            return 'Phase'

    def getGradTickSpacing(self, plotType):
        if plotType == 'Coherence':
            return (0.2, 0.1)
        else:
            return (60, 30)

    def getVarInfo(self):
        return (self.ui.boxA.currentText(), self.ui.boxB.currentText())

    def setVarParams(self, varPair):
        varA, varB = varPair
        self.ui.boxA.setCurrentText(varA)
        self.ui.boxB.setCurrentText(varB)

    def update(self):
        # Extract parameters from UI elements
        bw = self.ui.bwBox.value()
        logMode = True if self.ui.scaleModeBox.currentText() == 'Logarithmic' else False
        varA = self.ui.boxA.currentText()
        varB = self.ui.boxB.currentText()
        indexRng = self.getDataRange()
        shiftAmnt = self.ui.fftShift.value()
        interval = self.ui.fftInt.value()
        nPoints = abs(indexRng[1]-indexRng[0])
        detrendMode = self.ui.detrendCheck.isChecked()

        fftParam = (interval, shiftAmnt, bw)
        varPair = (varA, varB)

        if self.checkParameters(interval, shiftAmnt, bw, nPoints) == False:
            return

        # Calculate grid values for each plot and add to respective grids
        grids, freqs, times = self.calcGrids(indexRng, fftParam, varPair, detrendMode)
        self.cohPlt, self.phaPlt = self.generatePlots(grids, freqs, times, logMode)
        self.setupPlotLayout(self.cohPlt, 'Coherence', varPair, times, logMode)
        self.setupPlotLayout(self.phaPlt, 'Phase', varPair, times, logMode)

        # Enable exporting data
        fftParam = (interval, shiftAmnt, bw, detrendMode)
        for plt in [self.cohPlt, self.phaPlt]:
            exportFunc = functools.partial(self.exportData, self.window, plt,
                fftParam)
            plt.setExportEnabled(exportFunc)
        self.lastCalc = (freqs, times, grids[0], grids[1])

        if self.savedLineInfo: # Add any saved lines
            self.addSavedLine()
        elif len(self.lineInfoHist) > 0 and len(self.lineHistory) == 0:
            self.savedLineInfo = self.lineInfoHist
            self.lineInfoHist = []
            self.addSavedLine()
            self.savedLineInfo = None
    def setupPlotLayout(self, plt, plotType, varPair, times, logScaling):
        # Create gradient legend and add it to the graphics layout
        gradLegend = plt.getGradLegend(logMode=False)
        gradLegend.setBarWidth(38)

        cohMode = (plotType == 'Coherence')
        if cohMode:
            gradLegend.setTickSpacing(0.2, 0.1)
        else:
            gradLegend.setTickSpacing(60, 30)

        # Get labels
        title, axisLbl, legendLbl = self.getLabels(plotType, varPair, logScaling)
        plt.setTitle(title, size='13pt')
        plt.getAxis('left').setLabel(axisLbl)

        # Time range information
        timeInfo = self.getTimeRangeLbl(times[0], times[-1])

        # Determine which grid to use and add items to layout
        gridLt = self.ui.cohGrid if cohMode else self.ui.phaGrid
        gridLt.clear()
        gridLt.addItem(plt, 0, 0, 1, 1)
        gridLt.addItem(gradLegend, 0, 1, 1, 1)
        gridLt.addItem(legendLbl, 0, 2, 1, 1)
        gridLt.addItem(timeInfo, 1, 0, 1, 3)

    def getLabels(self, plotType, varInfos, logScaling):
        cohMode = (plotType == 'Coherence')
        varA, varB = varInfos
        title = 'Dynamic ' + plotType + ' Analysis'
        title = title + ' (' + varA + ' by ' + varB + ')'

        axisLbl = 'Frequency (Hz)'
        if logScaling:
            axisLbl = 'Log ' + axisLbl

        legendStrs = ['Angle', '[Degrees]']
        if cohMode:
            legendStrs = [plotType]
        legendLbl = StackedAxisLabel(legendStrs, angle=0)
        legendLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))

        return title, axisLbl, legendLbl

    def calcGrids(self, indexRng, fftParam, varPair, detrend=False):
        interval, shiftAmt, bw = fftParam
        varA, varB = varPair

        cohLst, phaLst = [], []
        timeSeries = []
        times = self.window.getTimes(varA, self.window.currentEdit)[0]

        en = self.window.currentEdit
        minIndex, maxIndex = indexRng
        startIndex, endIndex = minIndex, minIndex + interval
        while endIndex < maxIndex:
            # Save start time
            timeSeries.append(times[startIndex])
            # Calculate ffts and coh/pha
            N = endIndex - startIndex
            fft1 = self.getfft(varA, en, startIndex, endIndex, detrend=detrend)
            fft2 = self.getfft(varB, en, startIndex, endIndex, detrend=detrend)
            coh, pha = self.calculateCoherenceAndPhase(bw, fft1, fft2, N)
            cohLst.append(coh)
            phaLst.append(pha)
            # Move to next interval
            startIndex += shiftAmt
            endIndex = startIndex + interval
        timeSeries.append(times[startIndex]) # Add bounding end time
        timeSeries = np.array(timeSeries)

        # Transpose here to turn fft result rows into per-time columns
        cohGrid = np.array(cohLst).T
        phaGrid = np.array(phaLst).T
        freqs = self.calculateFreqList(bw, N)

        return (cohGrid, phaGrid), freqs, timeSeries

    def generatePlots(self, grids, freqs, times, logScaling):
        freqs = self.extendFreqs(freqs, logScaling)
        cohGrid, phaGrid = grids
        cohPlt = SpectrogramPlotItem(self.window.epoch, logScaling)
        phaPlt = PhaseSpectrogram(self.window.epoch, logScaling)
        cohRng = (0, 1.0)
        phaRng = (-180, 180)
        cohPlt.createPlot(freqs, cohGrid, times, cohRng, winFrame=self, 
            logColorScale=False)
        phaPlt.createPlot(freqs, phaGrid, times, phaRng, winFrame=self, 
            logColorScale=False)
        
        return (cohPlt, phaPlt)
    
    def extendFreqs(self, freqs, logScale):
        # Calculate frequency that serves as lower bound for plot grid
        diff = abs(freqs[1] - freqs[0])
        lowerFreqBnd = freqs[0] - diff
        if lowerFreqBnd == 0 and logScale:
            lowerFreqBnd = freqs[0] - diff/2
        freqs = np.concatenate([[lowerFreqBnd], freqs])
        return freqs

    def getDataRange(self):
        dstr = self.ui.boxA.currentText()

        # Get selection times and convert to corresponding data indices for dstr
        minTime, maxTime = self.window.getSelectionStartEndTimes()
        times = self.window.getTimes(dstr, 0)[0]
        startIndex = self.window.calcDataIndexByTime(times, minTime)
        endIndex = self.window.calcDataIndexByTime(times, maxTime)

        return startIndex, endIndex

    def getTimeRangeLbl(self, t1, t2):
        t1Str = self.window.getTimestampFromTick(t1)
        t2Str = self.window.getTimestampFromTick(t2)
        txt = 'Time Range: ' + t1Str + ' to ' + t2Str
        lbl = pg.LabelItem(txt)
        lbl.setAttr('justify', 'left')
        return lbl

    def showPointValue(self, yVal, tVal):
        if self.lastCalc is None:
            return

        freqs, times, cohGrid, phaGrid = self.lastCalc

        # Determine which tab is being displayed and adjust the values to be shown
        grid = cohGrid
        valType = 'Coherence'
        if self.ui.tabWidget.currentIndex() == 1:
            grid = phaGrid
            valType = 'Angle'

        prefix = 'Freq, ' + valType + ': '
        calcToShow = (times, freqs, grid)

        self.showValue(yVal, tVal, prefix, calcToShow)

    def addLineToPlot(self, line):
        if self.ui.tabWidget.currentIndex() == 1: # Phase mode
            self.phaPlt.addItem(line)
        else:
            self.cohPlt.addItem(line)
        self.lineHistory.add(line)

    def removeLinesFromPlot(self):
        if self.ui.tabWidget.currentIndex() == 1: # Phase mode
            plt = self.phaPlt
        else: # Coherence mode
            plt = self.cohPlt

        histCopy = self.lineHistory.copy()
        for line in histCopy:
            if line in plt.listDataItems():
                plt.removeItem(line)
                self.lineHistory.remove(line)