from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from FF_Time import FFTIME
from MagPy4UI import TimeEdit
from pyqtgraphExtensions import GridGraphicsLayout, LogAxis, MagPyAxisItem, DateAxis, GradientLegend

class SpectraLine(pg.PlotCurveItem):
    def __init__(self, freq, colors, times, *args, **kargs):
        self.freq = freq
        self.colors = colors
        self.times = times
        self.paths = None

        yVals = [freq]*len(times)
        pg.PlotCurveItem.__init__(self, x=times, y=yVals, *args, **kargs)

    def getPaths(self):
        # Creates a separate path for every square in plot
        if self.paths is None:
            x, y = self.getData()
            subPaths = []
            if x is None or len(x) == 0 or y is None or len(y) == 0:
                self.paths = [QtGui.QPainterPath()]
            else:
                timeLen = len(self.times)
                xPairs = [(self.times[i], self.times[i+1]) for i in range(0, timeLen-1)]
                yVal = y[0]
                for (x0, x1) in xPairs:
                    currPath = self.generatePath(np.array([x0, x1]), np.array([yVal, yVal]))
                    subPaths.append(currPath)
                self.paths = subPaths

        return self.paths

    def paint(self, p, opt, widget):
        if self.xData is None or len(self.xData) == 0:
            return

        paths = self.getPaths()
        yVal = self.yData[0]

        # Draws filled rects for every point using designated colors
        for pairNum in range(0, len(self.colors)):
            x0 = self.times[pairNum]
            x1 = self.times[pairNum+1]
            path = paths[pairNum]
            p2 = QtGui.QPainterPath(path)
            p2.lineTo(x1, self.opts['fillLevel'])
            p2.lineTo(x0, self.opts['fillLevel'])
            p2.lineTo(x0, yVal)
            p2.closeSubpath()
            p.fillPath(p2, pg.mkBrush(pg.mkColor(self.colors[pairNum])))

class SpectrogramPlotItem(pg.PlotItem):
    def addItem(self, item, *args, **kargs):
        """
        Add a graphics item to the view box. 
        If the item has plot data (PlotDataItem, PlotCurveItem, ScatterPlotItem), it may
        be included in analysis performed by the PlotItem.
        """
        self.items.append(item)
        vbargs = {}
        self.vb.addItem(item, *args, **vbargs)
        name = None
        if hasattr(item, 'implements') and item.implements('plotData'):
            name = item.name()
            self.dataItems.append(item)

            params = kargs.get('params', {})
            self.itemMeta[item] = params
            self.curves.append(item)

class SpectrogramViewBox(pg.ViewBox):
    def addItem(self, item, ignoreBounds=False):
        if item.zValue() < self.zValue():
            item.setZValue(self.zValue()+1)
        scene = self.scene()
        if scene is not None and scene is not item.scene():
            scene.addItem(item)
        item.setParentItem(self.childGroup)
        self.addedItems.append(item)

    def clear(self):
        for item in self.addedItems:
            self.scene().removeItem(item)
        self.addedItems = []

class DynamicSpectraUI(object):
    def setupUI(self, Frame, window):
        maxSizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        Frame.setWindowTitle('Dynamic Spectrogram')
        Frame.resize(1000, 800)
        layout = QtWidgets.QGridLayout(Frame)

        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout(window)
        self.gview.setCentralItem(self.glw)
        self.glw.layout.setHorizontalSpacing(15)

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

        spacer = QtWidgets.QSpacerItem(10, 1)

        self.updateBtn = QtWidgets.QPushButton('Update')
        self.updateBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Set up FFT parameter widgets
        self.addPair(settingsLt, lbls[0], self.fftInt, 0, 0, 1, 1, fftIntTip)
        self.addPair(settingsLt, lbls[1], self.fftShift, 1, 0, 1, 1, shiftTip)
        self.addPair(settingsLt, lbls[2], self.bwBox, 2, 0, 1, 1)

        # Set up data variable parameters
        self.addPair(settingsLt, lbls[3], self.dstrBox, 0, 2, 1, 1)
        self.addPair(settingsLt, lbls[4], self.scaleModeBox, 1, 2, 1, 1, scaleTip)
        self.addPair(settingsLt, lbls[5], self.fftDataPts, 2, 2, 1, 1, dtaPtsTip)
        settingsLt.addItem(spacer, 3, 4, 1, 1)
        settingsLt.addWidget(self.updateBtn, 2, 6, 1, 1)

        layout.addWidget(settingsFrame, 0, 0, 1, 2)

        # Set up time edits at bottom of window
        timeFrame = QtWidgets.QFrame()
        timeFrame.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        timeLt = QtWidgets.QHBoxLayout(timeFrame)
        self.timeEdit = TimeEdit(QtGui.QFont())
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())
        timeLt.addWidget(self.timeEdit.start)
        timeLt.addWidget(self.timeEdit.end)

        layout.addWidget(timeFrame, 9, 0, 1, 1)

        # Initialize data values for spinboxes and combo boxes
        for plt in window.lastPlotStrings:
            for (dstr, en) in plt:
                self.dstrBox.addItem(dstr)

        # Set up plot scaling combo box options
        self.scaleModeBox.addItem('Logarithmic')
        self.scaleModeBox.addItem('Linear')

        # Plot setup
        self.initPlot()
        layout.addWidget(self.gview, 1, 0, 8, 2)

        # Status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum))
        layout.addWidget(self.statusBar, 9, 1, 1, 1)

        # Default settings
        self.bwBox.setValue(3)

    def initVars(self, window):
        # Initializes the max/min times
        minTime, maxTime = window.getTimeTicksFromTimeEdit(self.timeEdit)
        times = window.getTimes(self.dstrBox.currentText(), 0)[0]
        startIndex = window.calcDataIndexByTime(times, minTime)
        endIndex = window.calcDataIndexByTime(times, maxTime)
        self.fftDataPts.setText(str(endIndex-startIndex))

    def setupGradient(self, colorMap, colorPos):
        # Create gradient legend and add it to the graphics layout
        gradLegend = GradientLegend()
        self.glw.addItem(gradLegend, 1, 5, 4, 1)
        gradLegend.setMinimumWidth(75)

        # Initialize gradient bounded by gradLegend's view rect
        pos = gradLegend.boundingRect()
        gradient = QtGui.QLinearGradient(pos.topLeft(), pos.bottomLeft())

        # Create and set color gradient based on colormap
        colors = colorMap.getColors()
        colors = list(map(pg.mkColor, colors))
        colorLocs = [0, 0.25, 0.5, 0.75, 1]
        for color, loc in zip(colors, colorLocs):
            gradient.setColorAt(loc, color)
        gradLegend.setGradient(gradient)

        # Set labels corresponding to color level
        colorPos = list(map(np.log10, colorPos))
        minLog, maxLog = colorPos[0], colorPos[-1]
        locs = [(i-minLog)/(maxLog-minLog) for i in colorPos[1:-1]]
        labels = list(map(str, list(map(int, colorPos[1:-1]))))
        labelsDict = {}
        for lbl, loc in zip(labels, locs):
            labelsDict[lbl] = loc
        gradLegend.setLabels(labelsDict)

        # Add in spacers to align gradient legend with plot top/bottom boundaries
        spacer = pg.LabelItem('') # Top spacer
        spacer.setMaximumHeight(self.plotItem.titleLabel.height())
        self.glw.addItem(spacer, 0, 5, 1, 1)

        spacer = pg.LabelItem('') # Bottom spacer
        botmAxisHt = self.plotItem.getAxis('bottom').boundingRect().height()
        spacer.setMaximumHeight(botmAxisHt*2)
        self.glw.addItem(spacer, 5, 5, 1, 1)

        # Add in legend labels and center them
        gradLabel = pg.LabelItem('Log Power')
        gradUnitsLbl = pg.LabelItem('(nT^2/Hz)')
        self.glw.addItem(gradLabel, 2, 6, 1, 1)
        self.glw.addItem(gradUnitsLbl, 3, 6, 1, 1)
        for lbl in [gradLabel, gradUnitsLbl]:
            lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.glw.layout.setColumnAlignment(6, QtCore.Qt.AlignCenter)

    def addPair(self, layout, name, elem, row, col, rowspan, colspan, tooltip=None):
        # Create a label for given widget and place both into layout
        lbl = QtWidgets.QLabel(name)
        lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        layout.addWidget(lbl, row, col, 1, 1)
        layout.addWidget(elem, row, col+1, rowspan, colspan)

        # Set any tooltips if given
        if tooltip is not None:
            lbl.setToolTip(tooltip)

    def initPlot(self):
        # Sets up plot settings / appearance
        self.glw.clear()

        # Create viewbox and custom axis items
        vb = SpectrogramViewBox()
        vb.enableAutoRange(x=False, y=False)
        dateAxis = DateAxis('bottom')
        if self.scaleModeBox.currentText() == 'Logarithmic':
            leftAxis = LogAxis(True, True, True, orientation='left')
        else:
            leftAxis = MagPyAxisItem(orientation='left')
        axisItems = {'bottom':dateAxis, 'left':leftAxis}

        self.plotItem = SpectrogramPlotItem(viewBox=vb, axisItems=axisItems)

        # Set title and lower downsampling
        dstrTitle = self.dstrBox.currentText()
        self.plotItem.setTitle('Dynamic Spectra Analysis'+' ('+dstrTitle+')', size='13pt')
        self.plotItem.setDownsampling(mode='subsample')

        # Shift axes ticks outwards instead of inwards
        la = self.plotItem.getAxis('left')
        maxTickLength = la.style['tickLength'] - 4
        la.setStyle(tickLength=maxTickLength*(-1))

        ba = self.plotItem.getAxis('bottom')
        maxTickLength = ba.style['tickLength'] - 2
        ba.setStyle(tickLength=maxTickLength*(-1))
        ba.autoSIPrefix = False # Disable extra label used in tick offset

        # Hide tick marks on right/top axes
        self.plotItem.showAxis('right')
        self.plotItem.showAxis('top')
        ra = self.plotItem.getAxis('right')
        ta = self.plotItem.getAxis('top')
        ra.setStyle(showValues=False, tickLength=0)
        ta.setStyle(showValues=False, tickLength=0)

        # Draw axes on top of any plot items (covered by SpectraLines o/w)
        for ax in [ba, la, ta, ra]:
            ax.setZValue(1000)

        # Disable mouse panning/scaling
        self.plotItem.setMouseEnabled(y=False, x=False)

        self.glw.addItem(self.plotItem, 0, 0, 6, 4)

    def addTimeInfo(self, timeRng, window):
        # Convert time ticks to tick strings
        startTime, endTime = timeRng
        startStr = str(FFTIME(startTime, Epoch=window.epoch).UTC)
        endStr = str(FFTIME(endTime, Epoch=window.epoch).UTC)

        # Remove day of year
        startStr = startStr[:4] + startStr[8:]
        endStr = endStr[:4] + endStr[8:]

        # Create time label widget and add to grid layout
        timeLblStr = 'Time Range: ' + startStr + ' to ' + endStr
        self.timeLbl = pg.LabelItem(timeLblStr)
        self.timeLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.glw.addItem(self.timeLbl, 6, 0, 1, 1)

class DynamicSpectra(QtGui.QFrame, DynamicSpectraUI):
    def __init__(self, window, parent=None):
        super(DynamicSpectra, self).__init__(parent)
        self.ui = DynamicSpectraUI()
        self.window = window
        self.wasClosed = False

        self.ui.setupUI(self, window)
        self.ui.updateBtn.clicked.connect(self.update)
        self.ui.timeEdit.start.dateTimeChanged.connect(self.updateParameters)
        self.ui.timeEdit.end.dateTimeChanged.connect(self.updateParameters)

    def closeEvent(self, ev):
        self.window.endGeneralSelect()
        self.window.statusBar.clearMessage()
        self.wasClosed = True

    def updateParameters(self):
        # Set num data points
        self.ui.initVars(self.window)
        nPoints = int(self.ui.fftDataPts.text())

        # Set interval and interval max
        self.ui.fftInt.setMaximum(nPoints)
        interval = max(min(nPoints, 10), int(nPoints*0.025))
        self.ui.fftInt.setValue(interval)

        # Set overlap and overlap max
        self.ui.fftShift.setMaximum(nPoints)
        overlap = int(interval/4)
        self.ui.fftShift.setValue(overlap)

    def checkParameters(self, interval, overlap, bw, numPoints):
        if interval <= overlap:
            self.ui.statusBar.showMessage('Error: Interval <= Shift amount')
            return False
        elif bw % 2 == 0:
            self.ui.statusBar.showMessage('Error: Bandwidth must be odd.')
            return False
        elif numPoints <= interval:
            self.ui.statusBar.showMessage('Error: Total num points <= Interval')
            return False
        return True

    def getDataRange(self):
        dstr = self.ui.dstrBox.currentText()

        # Get selection times and convert to corresponding data indices for dstr
        minTime, maxTime = self.window.getSelectionStartEndTimes()
        times = self.window.getTimes(dstr, 0)[0]
        startIndex = self.window.calcDataIndexByTime(times, minTime)
        endIndex = self.window.calcDataIndexByTime(times, maxTime)

        return startIndex, endIndex

    def update(self):
        # Determine data indices from lines
        dataRng = self.getDataRange()
        numPoints = dataRng[1] - dataRng[0]

        # Extract user-set parameters
        interval = self.ui.fftInt.value()
        shift = self.ui.fftShift.value()
        dstr = self.ui.dstrBox.currentText()
        bw = self.ui.bwBox.value()

        # Error checking for user parameters
        if self.checkParameters(interval, shift, bw, numPoints) == False:
            return

        # Initialize empty plot
        self.ui.statusBar.showMessage('Clearing previous plot...')
        self.ui.initPlot()

        # Generate plot grid and spectrogram from this
        self.calculate(dataRng, numPoints, interval, shift, bw, dstr)

    def calculate(self, dataRng, numPoints, interval, shift, bw, dstr):
        shiftAmnt = shift
        minIndex, maxIndex = dataRng
        startIndex, endIndex = minIndex, minIndex + interval

        self.ui.statusBar.showMessage('Calculating...')

        # Generate the list of powers and times associated w/ each FFT interval
        powerLst = []
        timeSeries = []
        times = self.window.getTimes(dstr, 0)[0]
        while endIndex < maxIndex:
            # Save start time
            timeSeries.append(times[startIndex])
            # Calculate freqs and powers
            freqs, powers = self.calcSpectra(dstr, bw, startIndex, endIndex)
            powerLst.append(np.array(powers))
            # Move to next interval
            startIndex += shiftAmnt
            endIndex = startIndex + interval
        timeSeries.append(times[startIndex])
        timeSeries = np.array(timeSeries)

        # Transpose here to turn fft result rows into per-time columns
        pixelGrid = np.array(powerLst)
        pixelGrid = pixelGrid.T

        # Map powers to a color gradient based on min/max values in time series
        maxPower = np.max(pixelGrid)
        minPower = np.min(pixelGrid)
        pixelGrid = self.mapValueToColor(pixelGrid, minPower, maxPower)

        self.ui.statusBar.showMessage('Generating plot...')
        self.createPlot(pixelGrid, freqs, timeSeries, shiftAmnt)

    def calcSpectra(self, dstr, bw, start, end):
        """
        Calculate the spectra for the given sub interval
        """
        # Convert time range to indices
        i0, i1 = start, end
        en = 0
        N = abs(i1-i0)

        fft = self.getfft(dstr, en, i0, i1)
        power = self.calculatePower(bw, fft, N)
        freqs = self.calculateFreqList(bw, N)
        return freqs, power

    def createPlot(self, pixelGrid, freqs, times, interval):
        gridHeight, gridWidth, gridColorDepth = pixelGrid.shape
        currProg = 0
        progStep = 100/gridHeight

        # Frequency that serves as lower bound for plot grid
        lowerFreqBnd = freqs[0] - abs(freqs[1]-freqs[0])
        lastFreq = lowerFreqBnd

        # Convert all frequences to log scale if necessary
        if self.ui.scaleModeBox.currentText() == 'Logarithmic':
            freqs = np.log10(freqs)
            lastFreq = np.log10(lastFreq)
            lowerFreqBnd = lastFreq

        # Convert every row corresp. to a freq to a SpectraLine object and plot
        for freqNum in range(len(freqs)):
            # Get frequency value, its power time series, and the list of times
            freqVal = freqs[freqNum]
            freqTimeSeries = pixelGrid[freqNum,:]
            timeVals = times

            # Map its powers to colors and add the SpectraLine to the plot
            colors = list(map(pg.mkColor, freqTimeSeries))
            pdi = SpectraLine(freqVal, colors, timeVals, fillLevel=lastFreq)
            self.ui.plotItem.addItem(pdi)

            # Update progress in status bar
            currProg += progStep
            self.ui.statusBar.showMessage('Generating plot... ' + str(int(currProg)) + '%')

            lastFreq = freqVal

        self.ui.statusBar.showMessage('Adjusting plot...')
        self.adjustPlotItem([times[0], times[-1]], (lowerFreqBnd, freqs[-1]))
        self.ui.statusBar.clearMessage()

    def mapValueToColor(self, vals, minPower, maxPower):
        rgbBlue = (25, 0, 245)
        rgbBlueGreen = (0, 245, 245)
        rgbGreen = (143, 245, 38)
        rgbYellow = (245, 245, 0)
        rgbRed = (245, 0, 25)

        colors = [rgbBlue, rgbBlueGreen, rgbGreen, rgbYellow, rgbRed]
        minLog = np.log10(minPower)
        maxLog = np.log10(maxPower)

        # Add in some padding for gradient range
        minLog = minLog - abs(minLog * 0.01)
        maxLog = maxLog + abs(maxLog * 0.01)

        # Determine the non-log values the color map will use
        midPoint = (minLog+maxLog) / 2
        leftMidPoint = (minLog + midPoint) / 2
        rightMidPoint = (midPoint + maxLog) / 2
        logLevels = [minLog, leftMidPoint, midPoint, rightMidPoint, maxLog]
        logLevels = [10**level for level in logLevels]

        # Generate tick mark values for gradient
        lwrBnd = int(np.ceil(minLog))
        upperBnd = int(np.floor(maxLog))
        logTicks = [i for i in range(lwrBnd, upperBnd+1)]
        logTicks = [minLog] + logTicks + [maxLog]
        logTicks = [10**level for level in logTicks]

        # Map power values to colors in gradient
        colorMap = pg.ColorMap(logLevels, colors)
        colorVals = colorMap.map(vals)

        # Set up gradient widget
        self.ui.setupGradient(colorMap, logTicks)

        return colorVals

    def adjustPlotItem(self, xRange, yRange):
        # Set log mode and left axis labels
        if self.ui.scaleModeBox.currentText() == 'Logarithmic':
            self.ui.plotItem.setLogMode(y=True)
            self.ui.plotItem.getAxis('left').setLabel('Log Frequency (Hz)')
        else:
            la = self.ui.plotItem.getAxis('left')
            self.ui.plotItem.setLogMode(y=False)
            self.ui.plotItem.getAxis('left').setLabel('Frequency (Hz)')

        # Disable auto range because setting log mode re-enables it
        vb = self.ui.plotItem.getViewBox()
        vb.enableAutoRange(x=False, y=False)

        # Set bottom axis time ticks and axis label
        rng = xRange[1] - xRange[0]
        mode = self.window.getTimeLabelMode(rng)
        lbl = self.window.getTimeLabel(rng)
        self.updateTimeTicks(xRange[0], xRange[1], mode)
        self.ui.plotItem.getAxis('bottom').setLabel(lbl)
        self.ui.addTimeInfo(xRange, self.window)

        # Set axis ranges
        self.ui.plotItem.setXRange(xRange[0], xRange[1], 0.0)
        self.ui.plotItem.setYRange(yRange[0], yRange[1], 0.0)
        self.ui.plotItem.hideButtons() # Hide autoscale buttons

    def updateTimeTicks(self, minTime, maxTime, mode):
        timeAxis = self.ui.plotItem.getAxis('bottom')
        timeAxis.window = self.window
        timeAxis.timeRange = (minTime, maxTime)
        timeAxis.updateTicks(self.window, mode, (minTime, maxTime))

    def getCommonVars(self, bw, N):
        if bw % 2 == 0: # make sure its odd
            bw += 1
        kmo = int((bw + 1) * 0.5)
        nband = (N - 1) / 2
        half = int(bw / 2)
        nfreq = int(nband - bw + 1)
        return bw,kmo,nband,half,nfreq

    def calculateFreqList(self, bw, N):
        bw,kmo,nband,half,nfreq = self.getCommonVars(bw, N)
        nfreq = int(nband - half + 1) #try to match power length
        C = N * self.window.resolution
        freq = np.arange(kmo, nfreq) / C
        if len(freq) < 2:
            print('Proposed spectra plot invalid!\nFrequency list has less than 2 values')
            return None
        return freq

    def calculatePower(self, bw, fft, N):
        bw,kmo,nband,half,nfreq = self.getCommonVars(bw, N)
        C = 2 * self.window.resolution / N
        fsqr = [ft * ft for ft in fft]
        power = [0] * nfreq
        for i in range(nfreq):
            km = kmo + i
            kO = int(km - half)
            kE = int(km + half) + 1

            power[i] = sum(fsqr[kO * 2 - 1:kE * 2 - 1]) / bw * C
        return power

    def getfft(self, dstr, en, i0, i1):
        data = self.window.getData(dstr, en)[i0:i1]
        fft = fftpack.rfft(data.tolist())
        return fft

