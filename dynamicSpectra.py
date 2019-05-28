from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from MagPy4UI import TimeEdit, NumLabel
from pyqtgraphExtensions import GridGraphicsLayout, LogAxis, MagPyAxisItem, DateAxis, GradientLegend, StackedAxisLabel
import bisect
import functools
from mth import Mth

class SpectraBase(object):
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

    def getfft(self, dstr, en, i0, i1):
        data = self.window.getData(dstr, en)[i0:i1]
        fft = fftpack.rfft(data.tolist())
        return fft

    def calculateFreqList(self, bw, N):
        bw,kmo,nband,half,nfreq = self.getCommonVars(bw, N)
        nfreq = int(nband - half + 1) #try to match power length
        C = N * self.window.resolution
        freq = np.arange(kmo, nfreq) / C
        if len(freq) < 2:
            print('Proposed spectra plot invalid!\nFrequency list has less than 2 values')
            return None
        return freq

class SpectraLegend(GradientLegend):
    def __init__(self, offsets=(0, 2)):
        GradientLegend.__init__(self, offsets)

    def getTickVals(self, minVal, maxVal, maxTicks=6):
        # Use an axis item to generate a list of potential tick values
        ax = pg.AxisItem(orientation='right')
        ax.setRange(minVal, maxVal)
        tickLst = ax.tickValues(minVal, maxVal, 1000)
        # Create a dictionary for each step size / tick subset
        tickStepDict = {}
        for step, sublst in tickLst:
            tickStepDict[step] = sublst

        # Merge tick value lists until at least a certain number are in list
        count = 0
        tickVals = []
        for key, sublst in tickLst:
            if count >= 6:
                break
            count += len(sublst)
            tickVals = tickVals + sublst
        tickVals.sort()

        # Select a subset of ticks if too many were selected
        if len(tickVals) > maxTicks:
            tickVals = self.limitTicks(tickVals, maxTicks)
        return tickVals

    def limitTicks(self, ticks, numVals):
        stepSize = max(int((len(ticks))/numVals) - 1, 2)
        newTicks = [ticks[i] for i in range(0, len(ticks), stepSize)]
        return newTicks

    def setRange(self, colorLst, minVal, maxVal, logMode=False, cstmTicks=None,
        cstmColorPos=None):
        # Initialize gradient bounded by gradLegend's view rect
        pos = self.boundingRect()
        gradient = QtGui.QLinearGradient(pos.topLeft(), pos.bottomLeft())

        # Calculate the integers between minVal/maxVal to label on color bar
        if cstmTicks is not None:
            colorPos = cstmTicks
        elif logMode:
            lwrBnd = int(np.ceil(minVal))
            upperBnd = int(np.floor(maxVal))
            colorPos = [i for i in range(lwrBnd, upperBnd+1)]
        else:
            colorPos = self.getTickVals(minVal, maxVal)

        # Create and set color gradient brush based on colormap
        colors = list(map(pg.mkColor, colorLst))
        colorLocs = [0, 1/3, 0.5, 2/3, 1]
        if cstmColorPos:
            colorLocs = cstmColorPos
        for color, loc in zip(colors, colorLocs):
            gradient.setColorAt(loc, color)
        self.setGradient(gradient)

        # Set labels corresponding to color level
        locs = [(i-minVal)/(maxVal-minVal) for i in colorPos]
        labels = []
        for val in colorPos:
            if val == 0:
                txt = '0'
            elif logMode or cstmTicks:
                txt = str(val)
            else:
                # If in linear mode, use scientific notation string
                txt = np.format_float_scientific(val, precision=3, trim='0', 
                    pad_left=False, sign=False, exp_digits=0)
            labels.append(txt)
        labelsDict = {}
        for lbl, loc in zip(labels, locs):
            labelsDict[lbl] = loc
        self.setLabels(labelsDict)

class SpectraLine(pg.PlotCurveItem):
    def __init__(self, freq, colors, times, window=None, *args, **kargs):
        # Takes the y-values, mapped color values, and time ticks
        self.freq = freq
        self.colors = colors
        self.times = times
        # Used to update window's status bar w/ the clicked value if passed
        self.window = window

        yVals = [freq]*len(times)
        pg.PlotCurveItem.__init__(self, x=times, y=yVals, *args, **kargs)

    def paint(self, p, opt, widget):
        if self.xData is None or len(self.xData) == 0:
            return

        yVal = self.yData[0]

        # Draws filled rects for every point using designated colors
        for pairNum in range(0, len(self.colors)):
            # Create a rectangle path
            color = self.colors[pairNum]
            if color == [255, 255, 255]:
                continue
            x0 = self.times[pairNum]
            x1 = self.times[pairNum+1]
            fillLevel = self.opts['fillLevel']
            pt1 = QtCore.QPointF(x0, yVal)
            p2 = QtGui.QPainterPath(pt1)
            p2.addRect(x0, fillLevel, x1-x0, yVal-fillLevel)
            # Get the corresponding color for the rectangle and fill w/ color
            p.fillPath(p2, color)

    def mouseClickEvent(self, ev):
        if self.window is None:
            return

        # If window argument was passed, show value in its status bar
        if ev.button() == QtCore.Qt.LeftButton:
            time = ev.pos().x()
            yVal = ev.pos().y()
            self.window.showPointValue(yVal, time)
            ev.accept()
        else:
            pg.PlotCurveItem.mouseClickEvent(self, ev)

class SpectrogramViewBox(pg.ViewBox):
    # Optimized viewbox class, removed some steps in addItem/clear methods
    def __init__(self, *args, **kargs):
        pg.ViewBox.__init__(self, *args, **kargs)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

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

class SpectrogramPlotItem(pg.PlotItem):
    def __init__(self, logMode=False):
        super(SpectrogramPlotItem, self).__init__(parent=None)
        self.logMode = logMode # Log scaling for y-axis parameter (Boolean)

        # Initialize colors for color map
        rgbBlue = (25, 0, 245)
        rgbBlueGreen = (0, 245, 245)
        rgbGreen = (127, 245, 0)
        rgbYellow = (245, 245, 0)
        rgbRed = (245, 0, 25)

        self.colors = [rgbBlue, rgbBlueGreen, rgbGreen, rgbYellow, rgbRed]
        self.valLevels = [] # Stores y-values corresp. to colors in self.colors

        # Create viewbox and set up custom axis items
        vb = SpectrogramViewBox()
        dateAxis = DateAxis('bottom')
        if self.logMode:
            leftAxis = LogAxis(True, True, True, orientation='left')
        else:
            leftAxis = MagPyAxisItem(orientation='left')
        axisItems = {'bottom':dateAxis, 'left':leftAxis}

        # Initialize default pg.PlotItem settings
        pg.PlotItem.__init__(self, viewBox=vb, axisItems=axisItems)

        # Set log/linear scaling after initialization
        if self.logMode:
            self.setLogMode(y=True)
        else:
            self.setLogMode(y=False)
        vb.enableAutoRange(x=False, y=False)

        self.plotSetup() # Additional plot appearance set up

    def plotSetup(self):
        # Shift axes ticks outwards instead of inwards
        la = self.getAxis('left')
        maxTickLength = la.style['tickLength'] - 4
        la.setStyle(tickLength=maxTickLength*(-1))

        ba = self.getAxis('bottom')
        maxTickLength = ba.style['tickLength'] - 2
        ba.setStyle(tickLength=maxTickLength*(-1))
        ba.autoSIPrefix = False # Disable extra label used in tick offset

        # Hide tick marks on right/top axes
        self.showAxis('right')
        self.showAxis('top')
        ra = self.getAxis('right')
        ta = self.getAxis('top')
        ra.setStyle(showValues=False, tickLength=0)
        ta.setStyle(showValues=False, tickLength=0)

        # Draw axes on top of any plot items (covered by SpectraLines o/w)
        for ax in [ba, la, ta, ra]:
            ax.setZValue(1000)

        # Disable mouse panning/scaling
        self.setMouseEnabled(y=False, x=False)
        self.setDownsampling(mode='subsample')
        self.hideButtons()

    def mkRGBColor(self, rgbVals):
        return QtGui.QColor(rgbVals[0], rgbVals[1], rgbVals[2])

    def mapValueToColor(self, vals, minPower, maxPower, logColorScale):
        if logColorScale:
            minLog = np.log10(minPower)
            maxLog = np.log10(maxPower)
        else:
            minLog, maxLog = minPower, maxPower

        # Determine the non-log values the color map will use
        midPoint = (minLog + maxLog) / 2
        oneThird = (maxLog - minLog) / 3
        logLevels = [minLog, minLog+oneThird, midPoint, maxLog-oneThird, maxLog]
        self.valLevels = logLevels

        # Map power values to colors (RGB values) according to gradient
        prevVals = vals[:]
        if logColorScale:
            cleanVals = vals.copy()
            cleanVals[cleanVals <= 0] = 1
            vals = np.log10(cleanVals) # Map values to log 10 first
        colorMap = pg.ColorMap(logLevels, self.colors)
        mappedVals = colorMap.map(vals)
        if logColorScale: # Map 0 values to white for log scaling
            mappedVals[prevVals<=0] = [255, 255, 255]

        return mappedVals

    def getGradLegend(self, logMode=True, offsets=None, cstmTicks=None):
        # Create color gradient legend based on color map
        minVal, maxVal = self.valLevels[0], self.valLevels[-1]
        if offsets:
            gradLegend = SpectraLegend(offsets)
        else:
            gradLegend = SpectraLegend()
        gradLegend.setRange(self.colors, minVal, maxVal, logMode, cstmTicks=cstmTicks)

        return gradLegend

    def updateTimeTicks(self, window, minTime, maxTime, mode):
        timeAxis = self.getAxis('bottom')
        timeAxis.window = window
        timeAxis.timeRange = (minTime, maxTime)
        timeAxis.updateTicks(window, mode, (minTime, maxTime))

    def addItem(self, item, *args, **kargs):
        # Optimized from original pg.PlotItem's addItem method
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

    def getSpectraLine(self, yVal, colors, times, winFrame, lastVal):
        pdi = SpectraLine(yVal, colors, times, winFrame, fillLevel=lastVal)
        return pdi

    # Takes the y-vals (length m), time ticks (length n), a matrix of values 
    # (of shape (m-1) x (n-1)), and a tuple of min/max values repres. by color gradient
    def createPlot(self, yVals, valueGrid, timeVals, colorRng, logColorScale=True, 
        winFrame=None, statusStrt=0, statusEnd=100):
        # Map values in grid to RGB colors
        minPower, maxPower = colorRng
        mappedGrid = self.mapValueToColor(valueGrid, minPower, maxPower, logColorScale)

        # Frequency/value that serves as lower bound for plot grid
        lastVal = yVals[0]

        # Convert all frequences to log scale if necessary
        if self.logMode:
            yVals = np.log10(yVals)
            lastVal = np.log10(lastVal)

        stepSize = (statusEnd-statusStrt) / (len(yVals) - 1)
        currentStep = statusStrt

        # Creates a SpectraLine object for every row in value grid
        for rowIndex in range(0, len(yVals)-1):
            yVal = yVals[rowIndex+1]
            colors = list(map(self.mkRGBColor, mappedGrid[rowIndex,:]))
            pdi = self.getSpectraLine(yVal, colors, timeVals, winFrame, lastVal)
            self.addItem(pdi)

            if winFrame: # If winFrame is passed, update progress in status bar
                currentStep += stepSize
                winFrame.ui.statusBar.showMessage('Generating plot...' + str(int(currentStep)) + '%')

            lastVal = yVal

class PhaseSpectrogram(SpectrogramPlotItem):
    def __init__(self, logMode=True):
        super(SpectrogramPlotItem, self).__init__(parent=None)
        SpectrogramPlotItem.__init__(self, logMode)
        rgbPink = (225, 0, 255)
        self.colors = [(225, 0, 255)] + self.colors + [(225, 0, 255)]
        self.colorPlacements = []
        centerTotal = 5/6
        startStop = 1/12
        cstmColPos = [0]
        for pos in [0, 1/3, 1/2, 2/3, 1]:
            currStop = startStop + (centerTotal*pos)
            cstmColPos.append(currStop)
        cstmColPos.append(1)
        self.colorPlacements = cstmColPos


    def mapValueToColor(self, valueGrid, minPower, maxPower, logColorScale):
        minVal, maxVal = -180, 180
        stepSize = 360 / 6
        colorPos = [(-180 + (360 * pos)) for pos in self.colorPlacements]
        colorMap = pg.ColorMap(colorPos, self.colors)
        mappedGrid = colorMap.map(valueGrid)
        return mappedGrid

    def getGradLegend(self, colors=None, cstmColPos=None, cstmTicks=None):
        top, bot = (31, 48)
        # Create color gradient legend based on color map
        minVal, maxVal = -180, 180
        cstmTicks = [i for i in range(minVal, maxVal+1, 60)]
        gradLegend = SpectraLegend((top, bot))
        gradLegend.setRange(self.colors, minVal, maxVal, cstmColorPos=self.colorPlacements,
            cstmTicks=cstmTicks, logMode=False)

        return gradLegend

class DynamicSpectraUI(object):
    def setupUI(self, Frame, window):
        maxSizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        Frame.setWindowTitle('Dynamic Spectrogram')
        Frame.resize(1100, 900)
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

        # Set up power min/max checkbox layout
        powerRngLt = self.setPowerRangeLt()
        self.powerRngSelectToggled(False)
        settingsLt.addLayout(powerRngLt, 0, 5, 3, 1)

        spacer = QtWidgets.QSpacerItem(10, 1)
        settingsLt.addItem(spacer, 2, 6, 1, 1)
        settingsLt.addWidget(self.updateBtn, 2, 7, 1, 1)

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
        itemSet = set()
        for plt in window.lastPlotStrings:
            for (dstr, en) in plt:
                if dstr not in itemSet: # Skip duplicates
                    self.dstrBox.addItem(dstr)
                itemSet.add(dstr)

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
        numPoints = abs(endIndex-startIndex)
        self.fftDataPts.setText(str(numPoints))

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

    def setPowerRangeLt(self):
        self.selectToggle = QtWidgets.QCheckBox(' Set Power Range: ')
        rangeLt = QtWidgets.QGridLayout()

        rngTip = 'Toggle to set max/min power levels represented by color gradient'
        self.selectToggle.setToolTip(rngTip)

        minTip = 'Minimum power level represented by color gradient'
        maxTip = 'Maximum power level represented by color gradient'

        self.powerMin = QtWidgets.QDoubleSpinBox()
        self.powerMax = QtWidgets.QDoubleSpinBox()

        # Set spinbox defaults
        for box in [self.powerMax, self.powerMin]:
            box.setPrefix('10^')
            box.setMinimum(-100)
            box.setMaximum(100)
            box.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        spc = '       ' # Spaces that keep spinbox lbls aligned w/ chkbx lbl

        rangeLt.addWidget(self.selectToggle, 0, 0, 1, 2)
        self.maxLbl = self.addPair(rangeLt, spc+'Max: ', self.powerMax, 1, 0, 1, 1, maxTip)
        self.minLbl = self.addPair(rangeLt, spc+'Min: ', self.powerMin, 2, 0, 1, 1, minTip)

        # Connects checkbox to func that enables/disables rangeLt's items
        self.selectToggle.toggled.connect(self.powerRngSelectToggled)
        return rangeLt

    def initPlot(self):
        # Clear previous plot
        self.glw.clear()

        # Set title and lower downsampling
        dstrTitle = self.dstrBox.currentText()

        logMode = self.scaleModeBox.currentText() == 'Logarithmic'
        self.plotItem = SpectrogramPlotItem(logMode=logMode)
        self.plotItem.setTitle('Dynamic Spectra Analysis'+' ('+dstrTitle+')', size='13pt')
        self.plotItem.setDownsampling(mode='subsample')

        self.glw.addItem(self.plotItem, 0, 0, 6, 4)

    def addTimeInfo(self, timeRng, window):
        # Convert time ticks to tick strings
        startTime, endTime = timeRng
        startStr = window.getTimestampFromTick(startTime)
        endStr = window.getTimestampFromTick(endTime)

        # Remove day of year
        startStr = startStr[:4] + startStr[8:]
        endStr = endStr[:4] + endStr[8:]

        # Create time label widget and add to grid layout
        timeLblStr = 'Time Range: ' + startStr + ' to ' + endStr
        self.timeLbl = pg.LabelItem(timeLblStr)
        self.timeLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.glw.addItem(self.timeLbl, 6, 0, 1, 1)

    def setupGradient(self):
        # Create gradient legend and add it to the graphics layout
        gradLegend = self.plotItem.getGradLegend()
        self.glw.addItem(gradLegend, 1, 5, 4, 1)
        gradLegend.setMinimumWidth(75)

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

    def powerRngSelectToggled(self, val):
        self.powerMax.setEnabled(val)
        self.powerMin.setEnabled(val)
        self.minLbl.setEnabled(val)
        self.maxLbl.setEnabled(val)

class DynamicSpectra(QtGui.QFrame, DynamicSpectraUI, SpectraBase):
    def __init__(self, window, parent=None):
        super(DynamicSpectra, self).__init__(parent)
        SpectraBase.__init__(self)
        self.ui = DynamicSpectraUI()
        self.window = window
        self.wasClosed = False

        self.gradRange = None # Custom color gradient range
        self.lastCalc = None # Previously calculated values, if any

        self.ui.setupUI(self, window)
        self.ui.updateBtn.clicked.connect(self.update)
        self.ui.timeEdit.start.dateTimeChanged.connect(self.updateParameters)
        self.ui.timeEdit.end.dateTimeChanged.connect(self.updateParameters)

    def closeEvent(self, ev):
        self.window.endGeneralSelect()
        self.window.clearStatusMsg()
        self.wasClosed = True

    def updateParameters(self):
        # Set num data points
        self.ui.initVars(self.window)
        nPoints = int(self.ui.fftDataPts.text())

        # Set interval and overlap max
        self.ui.fftInt.setMaximum(nPoints)
        self.ui.fftShift.setMaximum(nPoints)

        interval = max(min(nPoints, 10), int(nPoints*0.025))
        self.ui.fftInt.setValue(interval)
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

        if self.ui.selectToggle.isChecked():
            minGradPower = self.ui.powerMin.value()
            maxGradPower = self.ui.powerMax.value()
            # Adjust order if flipped
            minGradPower = min(minGradPower, maxGradPower)
            maxGradPower = max(minGradPower, maxGradPower)
            self.gradRange = (10**minGradPower, 10**maxGradPower)
        else:
            self.gradRange = None

        # Error checking for user parameters
        if self.checkParameters(interval, shift, bw, numPoints) == False:
            return

        # Initialize empty plot
        self.ui.statusBar.showMessage('Clearing previous plot...')
        self.ui.initPlot()

        # Generate plot grid and spectrogram from this
        self.calculate(dataRng, interval, shift, bw, dstr)

    def calculate(self, dataRng, interval, shift, bw, dstr):
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

        # Store calculations for displaying values at a point
        self.lastCalc = (timeSeries, freqs, pixelGrid)

        # Map powers to a color gradient based on min/max values in time series
        maxPower = np.max(pixelGrid)
        minPower = np.min(pixelGrid)
        if self.gradRange is not None and self.wasClosed == False:
            minPower, maxPower = self.gradRange # User-set range
        else:
            self.gradRange = None

        self.ui.statusBar.showMessage('Generating plot...')
        self.createPlot(pixelGrid, freqs, timeSeries, (minPower, maxPower))

    def calcSpectra(self, dstr, bw, start, end):
        """
        Calculate the spectra for the given sub interval
        """
        # Convert time range to indices
        i0, i1 = start, end
        en = self.window.currentEdit
        N = abs(i1-i0)

        fft = self.getfft(dstr, en, i0, i1)
        power = self.calculatePower(bw, fft, N)
        freqs = self.calculateFreqList(bw, N)
        return freqs, power

    def createPlot(self, pixelGrid, freqs, times, colorRng):
        # Frequency that serves as lower bound for plot grid
        diff = abs(freqs[1] - freqs[0])
        lowerFreqBnd = freqs[0] - diff
        if lowerFreqBnd == 0 and self.ui.scaleModeBox.currentText() == 'Logarithmic':
            lowerFreqBnd = freqs[0] - diff/2
        freqs = np.concatenate([[lowerFreqBnd], freqs])

        # Pass calculated values to SpectogramPlotItem to generate plot
        self.ui.plotItem.createPlot(freqs, pixelGrid, times, colorRng, winFrame=self)

        # Update x/y range and axis labels
        self.ui.statusBar.showMessage('Adjusting plot...')
        self.adjustPlotItem([times[0], times[-1]], (freqs[0], freqs[-1]))
        self.ui.statusBar.clearMessage()

    def adjustPlotItem(self, xRange, yRange):
        # Set log mode and left axis labels
        if self.ui.scaleModeBox.currentText() == 'Logarithmic':
            self.ui.plotItem.getAxis('left').setLabel('Log Frequency (Hz)')
            yMin, yMax = yRange
            yRange = (np.log10(yMin), np.log10(yMax))
        else:
            la = self.ui.plotItem.getAxis('left')
            self.ui.plotItem.getAxis('left').setLabel('Frequency (Hz)')

        # Disable auto range because setting log mode enables it
        vb = self.ui.plotItem.getViewBox()
        vb.enableAutoRange(x=False, y=False)

        # Set bottom axis time ticks and axis label
        rng = xRange[1] - xRange[0]
        mode = self.window.getTimeLabelMode(rng)
        lbl = self.window.getTimeLabel(rng)
        self.ui.plotItem.updateTimeTicks(self.window, xRange[0], xRange[1], mode)
        self.ui.plotItem.getAxis('bottom').setLabel(lbl)
        self.ui.addTimeInfo(xRange, self.window)

        # Set axis ranges
        self.ui.plotItem.setXRange(xRange[0], xRange[1], 0.0)
        self.ui.plotItem.setYRange(yRange[0], yRange[1], 0.0)
        self.ui.setupGradient()

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

    def showPointValue(self, freq, time):
        # Takes x,y values and uses them to find/display the power value
        if self.lastCalc is None:
            return

        if self.ui.scaleModeBox.currentText() == 'Logarithmic':
            freq = 10 ** freq

        # Find grid indices corresponding to the point
        times, freqs, powerGrid = self.lastCalc
        numRows, numCols = powerGrid.shape
        freqIndex = max(bisect.bisect(freqs, freq), 0)
        timeIndex = max(bisect.bisect(times, time)-1, 0)

        if freq > freqs[-1] or time < times[0] or time > times[-1]:
            self.ui.statusBar.clearMessage()
            return

        # Extract the grid's frequency and power values
        val = powerGrid[freqIndex][timeIndex]

        # Create and display the freq/power values in the status bar
        freqStr = NumLabel.formatVal(freq, 5)
        valStr = NumLabel.formatVal(val, 5)
        msg = 'Freq, Power: '+'('+freqStr+', '+valStr+')'
        self.ui.statusBar.showMessage(msg)

class DynamicCohPhaUI(object):
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
        timeLt = QtWidgets.QHBoxLayout()
        self.timeEdit = TimeEdit(QtGui.QFont())
        minDt, maxDt = window.getMinAndMaxDateTime()
        self.timeEdit.setupMinMax((minDt, maxDt))
        timeLt.addWidget(self.timeEdit.start)
        timeLt.addWidget(self.timeEdit.end)

        self.statusBar = QtWidgets.QStatusBar()
        timeLt.addWidget(self.statusBar)

        layout.addLayout(timeLt, 2, 0, 1, 2)

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

    def getSpacerItem(self):
        spacer = QtWidgets.QSpacerItem(10, 1, QSizePolicy.Minimum, QSizePolicy.Minimum)
        return spacer

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
            glw.layout.setHorizontalSpacing(15)

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

        tips = [fftIntTip, shiftTip, '']
        boxes = [self.fftInt, self.fftShift, self.bwBox]
        for i in range(0, 3):
            self.addPair(layout, lbls[i], boxes[i], i, 0, 1, 1, tips[i])
        layout.addItem(self.getSpacerItem(), 0, 2, 3, 1)

        # Set up frequency scaling mode box
        self.scaleBox = QtWidgets.QComboBox()
        self.scaleBox.addItem('Logarithmic')
        self.scaleBox.addItem('Linear')
        self.addPair(layout, 'Scaling mode: ', self.scaleBox, 1, 3, 1, 1, scaleTip)

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
        self.fftPoints = QtWidgets.QLabel()
        ptsTip = 'Total number of data points within selected time range'
        self.addPair(layout, 'Num Points: ', self.fftPoints, 2, 3, 1, 1, ptsTip)

        self.bwBox.setValue(3)
        self.bwBox.setSingleStep(2)
        self.bwBox.setMinimum(1)

        self.updtBtn = QtWidgets.QPushButton(' Update ')
        self.updtBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Column of spacers before update button
        for row in range(0, 3):
            spacer = self.getSpacerItem()
            layout.addItem(spacer, row, 6, 1, 1)

        layout.addWidget(self.updtBtn, 1, 7, 1, 1)
        return layout

class DynamicCohPha(QtGui.QFrame, DynamicCohPhaUI, SpectraBase):
    def __init__(self, window, parent=None):
        super(DynamicCohPha, self).__init__(parent)
        SpectraBase.__init__(self)
        self.ui = DynamicCohPhaUI()
        self.window = window
        self.wasClosed = False

        self.gradRange = None # Custom color gradient range
        self.lastCalc = None # Previously calculated values, if any

        # Get full set of dstrs currently plotted and sort them
        dstrs = set()
        for pltLst in self.window.lastPlotStrings:
            for dstr, en in pltLst:
                dstrs.add(dstr)
        dstrs = list(dstrs)
        dstrs.sort()

        # Close if nothing is currently being displayed
        if len(dstrs) == 0 or dstrs[0] == '':
            self.window.showStatusMsg('Error: Need at least one variable plotted.')
            self.close()

        self.ui.setupUI(self, window, dstrs)
        self.ui.updtBtn.clicked.connect(self.update)
        self.ui.timeEdit.start.dateTimeChanged.connect(self.updateParameters)
        self.ui.timeEdit.end.dateTimeChanged.connect(self.updateParameters)

    def closeEvent(self, ev):
        self.window.endGeneralSelect()
        self.window.clearStatusMsg()
        self.wasClosed = True
        self.close()

    def update(self):
        # Extract parameters from UI elements
        bw = self.ui.bwBox.value()
        logMode = True if self.ui.scaleBox.currentText() == 'Logarithmic' else False
        varA = self.ui.boxA.currentText()
        varB = self.ui.boxB.currentText()
        indexRng = self.getDataRange()
        shiftAmnt = self.ui.fftShift.value()
        interval = self.ui.fftInt.value()
        nPoints = abs(indexRng[1]-indexRng[0])

        if self.checkParameters(interval, shiftAmnt, bw, nPoints) == False:
            return

        self.calculate(varA, varB, bw, logMode, indexRng, shiftAmnt, interval)

    def calculate(self, varA, varB, bw, logMode, indexRng, shiftAmt, interval):
        cohLst, phaLst = [], []
        timeSeries = []
        times = self.window.getTimes(varA, 0)[0]

        self.ui.statusBar.showMessage('Calculating...')

        minIndex, maxIndex = indexRng
        startIndex, endIndex = minIndex, minIndex + interval
        while endIndex < maxIndex:
            # Save start time
            timeSeries.append(times[startIndex])
            # Calculate ffts and coh/pha
            N = endIndex - startIndex + 1
            fft1 = self.getfft(varA, 0, startIndex, endIndex)
            fft2 = self.getfft(varB, 0, startIndex, endIndex)
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

        # Save calculated values
        self.lastCalc = (freqs, timeSeries, cohGrid, phaGrid)

        # Get lower bound for frequencies and add to beginning of freq list
        diff = freqs[1] - freqs[0]
        lowerBnd = freqs[0] - diff
        if lowerBnd == 0 and logMode:
            lowerBnd = freqs[0] - diff/2
        freqs = np.concatenate([[lowerBnd], freqs])

        # Generate plots
        cohPlt, phaPlt = self.createPlots(freqs, timeSeries, cohGrid, phaGrid, logMode)

        # Adjust plot ranges and y axis labels
        timeRng = (timeSeries[0], timeSeries[-1])
        freqRng = (freqs[0], freqs[-1])
        self.adjustPlots(cohPlt, phaPlt, varA, varB, logMode, timeRng, freqRng)

        # Add in time label info at bottom
        for grid in [self.ui.cohGrid, self.ui.phaGrid]:
            timeLbl = self.getTimeRangeLbl(timeSeries[0], timeSeries[-1])
            grid.addItem(timeLbl, 1, 0, 1, 1)
        self.ui.statusBar.clearMessage()

    def createPlots(self, freqs, times, cohGrid, phaGrid, logMode):
        # Generate the color mapped plots from the value grids
        cohPlt = SpectrogramPlotItem(logMode)
        phaPlt = PhaseSpectrogram(logMode)
        cohRng = (0, 1.0)
        phaRng = (-180, 180)
        cohPlt.createPlot(freqs, cohGrid, times, cohRng, winFrame=self, 
            logColorScale=False, statusStrt=0, statusEnd=50)
        phaPlt.createPlot(freqs, phaGrid, times, phaRng, winFrame=self, 
            logColorScale=False, statusStrt=50, statusEnd=100)

        # Get color gradients
        cohTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cohGrad = cohPlt.getGradLegend(cstmTicks=cohTicks, logMode=False, 
            offsets=(31, 48))
        cohGrad.setFixedWidth(65)

        phaGrad = phaPlt.getGradLegend()
        phaGrad.setFixedWidth(60)

        # Get color bar labels
        cohLbl = pg.LabelItem('Coherence')
        cohLbl.setFixedWidth(65)

        phaLbl = StackedAxisLabel(['Phase', '[Degrees]'], angle=0)
        phaLbl.setFixedWidth(70)

        # Add items into grids
        for grid, plt, grad, lbl in zip([self.ui.cohGrid, self.ui.phaGrid],
            [cohPlt, phaPlt], [cohGrad, phaGrad], [cohLbl, phaLbl]):
            grid.clear()
            grid.addItem(plt)
            grid.nextCol()
            grid.addItem(grad)
            grad.updateWidth(35)
            grid.nextCol()
            grid.addItem(lbl)

        return cohPlt, phaPlt

    def adjustPlots(self, cohPlt, phaPlt, varA, varB, logMode, xRng, yRng):
        # Set titles
        subTitle = '(' + varA + ' by ' + varB + ')'
        cohPlt.setTitle('Dynamic Coherence Analysis ' + subTitle, size='13pt')
        phaPlt.setTitle('Dynamic Phase Analysis ' + subTitle, size='13pt')

        # Set time labels
        rng = xRng[1] - xRng[0]
        mode = self.window.getTimeLabelMode(rng)
        lbl = self.window.getTimeLabel(rng)
        if logMode:
            a = np.log10(yRng[0])
            b = np.log10(yRng[1])
            yRng = (a, b)

        # Update plot ranges and set axis labels
        for plt in [cohPlt, phaPlt]:
            plt.updateTimeTicks(self.window, xRng[0], xRng[1], mode)
            if logMode:
                plt.getAxis('left').setLabel('Log Frequency (Hz)')
            else:
                plt.getAxis('left').setLabel('Frequency (Hz)')
            plt.setXRange(xRng[0], xRng[1], 0)
            plt.setYRange(yRng[0], yRng[1], 0)
            plt.getAxis('bottom').setLabel(lbl)

    def getDataRange(self):
        dstr = self.ui.boxA.currentText()

        # Get selection times and convert to corresponding data indices for dstr
        minTime, maxTime = self.window.getSelectionStartEndTimes()
        times = self.window.getTimes(dstr, 0)[0]
        startIndex = self.window.calcDataIndexByTime(times, minTime)
        endIndex = self.window.calcDataIndexByTime(times, maxTime)

        return startIndex, endIndex

    def updateParameters(self):
        # Set num data points
        minTime, maxTime = self.window.getTimeTicksFromTimeEdit(self.ui.timeEdit)
        times = self.window.getTimes(self.ui.boxA.currentText(), 0)[0]
        startIndex = self.window.calcDataIndexByTime(times, minTime)
        endIndex = self.window.calcDataIndexByTime(times, maxTime)
        nPoints = abs(endIndex-startIndex)
        self.ui.fftPoints.setText(str(nPoints))

        # Set interval max and overlap max
        self.ui.fftInt.setMaximum(nPoints)
        self.ui.fftShift.setMaximum(nPoints)

        # Calculate some parameters to use for first generated plot
        interval = max(min(nPoints, 10), int(nPoints*0.025))
        overlap = int(interval/4)
        self.ui.fftInt.setValue(interval)
        self.ui.fftShift.setValue(overlap)

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

        if self.ui.scaleBox.currentText() == 'Logarithmic':
            yVal = 10 ** yVal

        # Find grid indices corresponding to the point
        freqIndex = max(bisect.bisect(freqs, yVal), 0)
        timeIndex = max(bisect.bisect(times, tVal)-1, 0)

        if yVal > freqs[-1] or tVal < times[0] or tVal > times[-1]:
            self.ui.statusBar.clearMessage()
            return

        # Determine which tab is being displayed and adjust the values to be shown
        grid = cohGrid
        valType = 'Coherence'
        if self.ui.tabWidget.currentIndex() == 1:
            grid = phaGrid
            valType = 'Angle'

        freqStr = NumLabel.formatVal(yVal, 5)
        mappedVal = str(NumLabel.formatVal(grid[freqIndex][timeIndex], 5))
        msg = 'Freq, ' + valType + ' = ' + '(' + freqStr + ', ' + mappedVal + ')'
        self.ui.statusBar.showMessage(msg)

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

    def calculateCoherenceAndPhase(self, bw, fft0, fft1, N):
        bw,kmo,nband,half,nfreq = self.getCommonVars(bw, N)
        kStart = kmo - half
        kSpan = half * 4 + 1

        csA = fft0[:-1] * fft1[:-1] + fft0[1:] * fft1[1:]
        qsA = fft0[:-1] * fft1[1:] - fft1[:-1] * fft0[1:]
        pAA = fft0[:-1] * fft0[:-1] + fft0[1:] * fft0[1:]
        pBA = fft1[:-1] * fft1[:-1] + fft1[1:] * fft1[1:]

        csSum = np.zeros(nfreq)
        qsSum = np.zeros(nfreq)
        pASum = np.zeros(nfreq)
        pBSum = np.zeros(nfreq)

        for n in range(nfreq):
            KO = (kStart + n) * 2 - 1
            KE = KO + kSpan

            csSum[n] = sum(csA[KO:KE:2])
            qsSum[n] = sum(qsA[KO:KE:2])
            pASum[n] = sum(pAA[KO:KE:2])
            pBSum[n] = sum(pBA[KO:KE:2])

        coh = (csSum * csSum + qsSum * qsSum) / (pASum * pBSum)
        pha = np.arctan2(qsSum, csSum) * Mth.R2D

        return coh,pha