from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from copy import copy

from .plotAppearance import DynamicPlotApp
import pyqtgraph as pg
from scipy import fftpack, signal
import numpy as np
from .MagPy4UI import TimeEdit, NumLabel, StackedAxisLabel
from .pyqtgraphExtensions import GridGraphicsLayout
from .plotBase import MagPyAxisItem, DateAxis, MagPyPlotItem
import bisect
import functools
from .mth import Mth
from .layoutTools import BaseLayout
from .simpleCalculations import ExpressionEvaluator
import os

class AddIcon(QtGui.QPixmap):
    ''' Pixmap renders arrow pointing to bottom right '''
    def __init__(self):
        QtGui.QPixmap.__init__(self, 20, 20)

        # Set up painter and background
        self.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(self)

        # Set pen
        pen = pg.mkPen((125, 125, 125))
        pen.setWidthF(2.5)
        pen.setJoinStyle(QtCore.Qt.MiterJoin)
        p.setPen(pen)
        mid = 10

        # Determine corner points
        topLeft = QtCore.QPointF(4, 4)
        bottomRight = QtCore.QPointF(20-2, 20-2)
        bottomLeft = QtCore.QPointF(6, 20-2)
        topRight = QtCore.QPointF(20-2, 6)

        # Draw arrow
        p.drawLine(topLeft, bottomRight)
        p.drawLine(bottomRight, bottomLeft)
        p.drawLine(bottomRight, topRight)

        p.end()

class AddBtn(QtWidgets.QPushButton):
    ''' Button showing arrow pointing to bottom right '''
    def __init__(self):
        QtWidgets.QPushButton.__init__(self)
        icon = QtGui.QIcon(AddIcon())
        self.setIcon(icon)
        self.setToolTip('Add plot to main plot grid')

class SpectraLineEditorUI(BaseLayout):
    def setupUI(self, Frame, window):
        Frame.resize(100, 100)
        Frame.setWindowTitle('Line Tool')
        layout = QtWidgets.QGridLayout(Frame)
        self.lineColor = '#000000'
        self.lineStyles = {'Solid': QtCore.Qt.SolidLine, 'Dashed':QtCore.Qt.DashLine, 
            'Dotted': QtCore.Qt.DotLine, 'DashDot': QtCore.Qt.DashDotLine}

        helpInfo = 'Please enter an expression to calculate and press \'Plot\':'
        layout.addWidget(QtWidgets.QLabel(helpInfo), 0, 0, 1, 4)

        # Set up text box for user to input expression
        self.textBox = QtWidgets.QTextEdit()
        exampleTxt = 'Examples:\nLine = (Bx^2 + By^2 + Bz^2)^(1/2) - 50 \n' +\
                'Line = 0.24'
        self.textBox.setPlaceholderText(exampleTxt)
        layout.addWidget(self.textBox, 1, 0, 1, 4)

        # Set up line appearance options
        linePropFrame = self.setupLineProperties()
        layout.addWidget(linePropFrame, 2, 0, 1, 4)

        # Set up state checkboxes
        self.fixLine = QtWidgets.QCheckBox('Fix Line')
        self.fixLine.setToolTip('Replot line after plot is updated')

        self.keepPrevLines = QtWidgets.QCheckBox('Keep Previous Lines')
        self.keepPrevLines.setToolTip('Keep previously plotted lines on plot')

        # Set up clear/plot buttons and status bar
        self.clearBtn = QtWidgets.QPushButton('Clear')
        self.addBtn = QtWidgets.QPushButton('Plot')
        self.statusBar = QtWidgets.QStatusBar()

        layout.addWidget(self.statusBar, 4, 0, 1, 2)
        layout.addWidget(self.fixLine, 3, 0, 1, 1)
        layout.addWidget(self.keepPrevLines, 3, 1, 1, 2)
        layout.addWidget(self.clearBtn, 4, 2, 1, 1)
        layout.addWidget(self.addBtn, 4, 3, 1, 1)

    def setupLineProperties(self):
        frame = QtWidgets.QGroupBox('Line Properties')
        layout = QtWidgets.QGridLayout(frame)

        self.colorBtn = QtWidgets.QPushButton()
        styleSheet = "* { background: #000000 }"
        self.colorBtn.setStyleSheet(styleSheet)
        self.colorBtn.setFixedWidth(45)

        self.lineStyle = QtWidgets.QComboBox()
        self.lineStyle.addItems(self.lineStyles.keys())

        self.lineWidth = QtWidgets.QSpinBox()
        self.lineWidth.setMinimum(1)
        self.lineWidth.setMaximum(5)

        for col, elem, name in zip([0, 2, 4], [self.colorBtn, self.lineStyle, 
            self.lineWidth], ['Color: ', ' Style: ', ' Width: ']):
            self.addPair(layout, name, elem, 0, col, 1, 1)
        return frame

class SpectraLineEditor(QtWidgets.QFrame, SpectraLineEditorUI):
    def __init__(self, spectraFrame, window, dataRange, parent=None):
        super().__init__(parent)
        self.spectraFrame = spectraFrame
        self.window = window
        self.plottedLines = []
        self.ui = SpectraLineEditorUI()
        self.ui.setupUI(self, window)

        self.ui.colorBtn.clicked.connect(self.openColorSelect)
        self.ui.clearBtn.clicked.connect(self.clearPlot)
        self.ui.addBtn.clicked.connect(self.addToPlot)
        if self.spectraFrame.savedLineInfo:
            self.ui.fixLine.setChecked(True)
        self.ui.fixLine.toggled.connect(self.fixedLineToggled)

    def createLine(self, dta, times, color, style, width):
        # Constructs a plotDataItem object froms given settings and data
        pen = pg.mkPen(color=color, width=width)
        pen.setStyle(style)
        line = pg.PlotDataItem(times, dta, pen=pen)
        return line

    def evalExpr(self, exprStr, sI, eI):
        # Attempt to evaluate expression, print error if an exception occurs
        try:
            rng = (sI, eI)
            name, dta = ExpressionEvaluator.evaluate(exprStr, self.window, rng)
            if dta is None:
                return None

            # Reshape constants to match times length
            if isinstance(dta, (float, int)):
                dta = [dta] * (eI-sI)

            # Adjust values if plot is in log scale
            if self.isLogMode():
                dta = np.log10(dta)

            return dta

        except:
            return None

    def addToPlot(self):
        if not self.keepMode():
            self.clearPlot()
        sI, eI = self.spectraFrame.getDataRange()
        exprStr = self.ui.textBox.toPlainText()
        if exprStr == '':
            return

        dta = self.evalExpr(exprStr, sI, eI)
        if dta is None: # Return + print error message for invalid expression
            self.ui.statusBar.showMessage('Error: Invalid operation')
            return

        # Extract line settings from UI and times to use for data
        arbDstr = self.window.DATASTRINGS[0]
        times = self.window.getTimes(arbDstr, 0)[0][sI:eI]
        color = self.ui.lineColor
        width = self.ui.lineWidth.value()
        lineStyle = self.ui.lineStyles[self.ui.lineStyle.currentText()]

        # Create line from user input and add it to the plot item
        lineItem = self.createLine(dta, times, color, lineStyle, width)
        self.spectraFrame.addLineToPlot(lineItem)
        self.spectraFrame.lineInfoHist.add(self.getLineInfo())
        self.saveLineInfo()

    def clearPlot(self):
        # TODO: Clear saved line info after plot is updated from main window
        self.spectraFrame.removeLinesFromPlot()
        self.clearLineInfo()

    def isLogMode(self):
        mode = self.spectraFrame.ui.scaleModeBox.currentText()
        if mode == 'Linear':
            return False
        return True

    def openColorSelect(self):
        clrDialog = QtWidgets.QColorDialog(self)
        clrDialog.show()
        clrDialog.colorSelected.connect(self.setButtonColor)

    def setButtonColor(self, color):
        styleSheet = "* { background:" + color.name() + " }"
        self.ui.colorBtn.setStyleSheet(styleSheet)
        self.ui.colorBtn.show()
        self.ui.lineColor = color.name()

    def keepMode(self):
        return self.ui.keepPrevLines.isChecked()

    def getLineInfo(self):
        expr = self.ui.textBox.toPlainText()
        color = self.ui.lineColor
        width = self.ui.lineWidth.value()
        style = self.ui.lineStyles[self.ui.lineStyle.currentText()]
        lineTuple = (expr, color, width, style)
        return lineTuple

    def saveLineInfo(self):
        # Extract parameters used to plot current line
        lineTuple = self.getLineInfo()

        if self.keepMode(): # Append to list if keeping previous lines on plot
            self.plottedLines.append(lineTuple)
        else: # Otherwise, set to current line only
            self.plottedLines = [lineTuple]

        if self.ui.fixLine.isChecked(): # Save in outer frame if fixLine box is checked
            self.fixedLineToggled(True)

    def clearLineInfo(self):
        self.plottedLines = []

    def fixedLineToggled(self, val):
        if val:
            self.spectraFrame.setSavedLine(self.plottedLines)
        else:
            self.spectraFrame.clearSavedLine()

class ColorBarAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyle(tickLength=-5)

    def tickSpacing(self, minVal, maxVal, size):
        dif = abs(maxVal - minVal)
        if dif == 0:
            return []
        vals = pg.AxisItem.tickSpacing(self, minVal, maxVal, size)

        # Adjust spacing to use 'neat' numbers as lower bound
        newVals = []
        for spacing, ofst in vals:
            lowerBound = int(minVal/spacing)*spacing + spacing
            newVals.append((spacing, lowerBound))
        return newVals

    def tickValues(self, minVal, maxVal, size):
        # Limit tick values to top level if sufficient, or add in
        # second level minor ticks and limit the number of ticks
        tickVals = pg.AxisItem.tickValues(self, minVal, maxVal, size)
        if len(tickVals) == 0:
            return []

        majorSpacing, majorTicks = tickVals[0]
        if len(majorTicks) >= 4:
            return [(majorSpacing, majorTicks)]
        elif len(tickVals) > 1:
            minorSpacing, minorTicks = tickVals[1]
            if len(majorTicks+minorTicks) >= 10:
                allTicks = majorTicks + minorTicks
                allTicks.sort()
                return [(majorSpacing*2, allTicks[::2])]
            else:
                return tickVals
        else:
            return tickVals

class ColorBar(pg.GraphicsWidget):
    def __init__(self, gradient, parent=None):
        super().__init__()
        self.gradient = gradient
        self.setMaximumWidth(50)
        self.setMinimumWidth(30)

    def setGradient(self, gradient):
        self.gradient = gradient

    def getGradient(self):
        return self.gradient

    def paint(self, p, opt, widget):
        ''' Fill the bounding rect w/ current gradient '''
        pg.GraphicsWidget.paint(self, p, opt,widget)
        # Get paint rect and pen
        rect = self.boundingRect()
        pen = pg.mkPen((0, 0, 0))
        pen.setJoinStyle(QtCore.Qt.MiterJoin)

        # Set gradient bounds
        self.gradient.setStart(0, rect.bottom())
        self.gradient.setFinalStop(0, rect.top())

        # Draw gradient
        p.setPen(pen)
        p.setBrush(self.gradient)
        p.drawRect(rect)

class GradLegend(pg.GraphicsLayout):
    def __init__(self, parent=None, *args, **kwargs):
        # Initialize state and legend elements
        self.valueRange = (0, 1)
        self.colorBar = ColorBar(QtGui.QLinearGradient())
        self.axisItem = ColorBarAxis(orientation='right')

        # Set default contents spacing/margins
        super().__init__(parent)
        self.layout.setVerticalSpacing(0)
        self.layout.setHorizontalSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))

        self.addItem(self.colorBar, 0, 0, 1, 1)
        self.addItem(self.axisItem, 0, 1, 1, 1)

    def getGradient(self):
        return self.colorBar.getGradient()

    def getValueRange(self):
        ''' Returns the numerical value range represented by gradient '''
        return self.valueRange

    def setRange(self, gradient, valRange):
        ''' Sets the color gradient and numerical value range it represents '''
        startVal, stopVal = valRange
        self.axisItem.setRange(startVal, stopVal)
        self.colorBar.setGradient(gradient)
        self.valueRange = valRange

    def setOffsets(self, top, bottom, left=None, right=None):
        ''' Set top/bottom margins and optionally, left and right margins '''
        if left is None:
            left = 10
        if right is None:
            right = 0
        self.layout.setContentsMargins(left, top, right, bottom)

    def setEdgeMargins(self, left, right):
        ''' Set left and right margins '''
        lm, tm, rm, bm = self.layout.getContentsMargins()
        self.layout.setContentsMargins(left, tm, right, bm)

    def setBarWidth(self, width):
        ''' Sets the width of the color bar/gradient to a fixed amount '''
        self.colorBar.setFixedWidth(width)
        self.layout.setColumnMaximumWidth(0, width)
        self.layout.setColumnMinimumWidth(0, width)

    def setTickSpacing(self, major, minor):
        self.axisItem.setTickSpacing(major, minor)

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

    def getfft(self, dstr, en, i0, i1, detrend=False):
        data = self.window.getData(dstr, en)[i0:i1]
        if detrend:
            data = signal.detrend(data)
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

    def splitfft(self, fft):
        '''
            Splits FFT results into its real and imaginary parts as separate
            lists
        '''
        fftReal = fft[1::2]
        fftImag = fft[2::2]
        return fftReal, fftImag

    def fftToComplex(self, fft):
        '''
            Converts fft (cos, sin) pairs into complex numbers
        '''
        rfft, ifft = self.splitfft(fft)
        cfft = np.array(rfft, dtype=np.complex)
        cfft = cfft[:len(ifft)]
        cfft.imag = ifft
        return cfft

    # Spectra calculations
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

    # Coherence and phase calculations
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

    # Wave analysis calculations
    def computeSpectralMats(self, cffts):
        # Computes the complex versions of the spectral matrices
        mats = np.zeros((len(cffts[0]), 3, 3), dtype=np.complex)
        for r in range(0, 3):
            for c in range(0, r+1):
                f0, f1 = cffts[r], cffts[c]
                res = f0 * np.conj(f1)
                conjRes = np.conj(res)
                mats[:,r][:,c] = res
                if r != c:
                    mats[:,c][:,r] = conjRes

        return mats

class DynamicAnalysisTool(SpectraBase):
    def __init__(self):
        self.lineTool = None
        self.maskTool = None
        self.savedLineInfo = None
        self.lineHistory = set()
        self.lineInfoHist = set()
        self.fftBins = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
            16384, 32768, 65536, 131072]
        self.fftBinBound = self.fftBins[-1]*4

    def getState(self):
        state = {}
        state['fftParams'] = self.getFFTParam()
        state['scaleMode'] = self.getAxisScaling()
        state['detrend'] = self.getDetrendMode()
        state['lineInfo'] = self.lineInfoHist
        state['varInfo'] = self.getVarInfo()
        state['gradRange'] = self.getGradRange()
        state['plotType'] = self.getToolType()

        return state

    def loadState(self, state):
        self.setParams(state['fftParams'], state['scaleMode'], state['detrend'])
        self.setGradRange(state['gradRange'])
        self.setVarParams(state['varInfo'])
        self.savedLineInfo = state['lineInfo']

    def setParams(self, fftParams, scale, detrend):
        # Set FFT parameters
        fftBoxes = [self.ui.fftInt, self.ui.fftShift, self.ui.bwBox]
        for box, val in zip(fftBoxes, list(fftParams)):
            box.setValue(val)

        # Set detrend mode and plot scale
        self.setDetrendMode(detrend)
        self.setAxisScaling(scale)

    def setGradRange(self, rng):
        if rng is None:
            return

        # Check gradient range checkbox
        self.ui.selectToggle.setChecked(True)

        # Scale values according to spinbox type (linear vs log scale)
        minVal, maxVal = rng
        if self.ui.valueMin.prefix() == '10^':
            minVal, maxVal = np.log10(minVal), np.log10(maxVal)

        # Set min/max values
        self.ui.valueMin.setValue(minVal)
        self.ui.valueMax.setValue(maxVal)

    def setVarParams(self, varInfo):
        pass

    def getGradRange(self):
        return None

    def getFFTParam(self):
        interval = self.ui.fftInt.value()
        shift = self.ui.fftShift.value()
        bw = self.ui.bwBox.value()
        return (interval, shift, bw)

    def getAxisScaling(self):
        return self.ui.scaleModeBox.currentText()

    def getDetrendMode(self):
        return self.ui.detrendCheck.isChecked()
    
    def setDetrendMode(self, val):
        self.ui.detrendCheck.setChecked(val)

    def setAxisScaling(self, mode):
        self.ui.scaleModeBox.setCurrentText(mode)

    def getGradTickSpacing(self, plotType=None):
        return None

    def updateParameters(self):
        # Set num data points
        self.ui.initVars(self.window)
        nPoints = int(self.ui.fftDataPts.text())

        # Set interval and overlap max
        self.ui.fftInt.setMaximum(nPoints)
        self.ui.fftShift.setMaximum(nPoints)

        interval = max(min(nPoints, 10), int(nPoints*0.025))
        if nPoints > self.fftBins[4] and nPoints < self.fftBinBound:
            index = bisect.bisect(self.fftBins, interval)
            if index < len(self.fftBins):
                interval = self.fftBins[index]

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

    def getLabels(self, varInfo, logScale):
        pass # Y axis label, title, stackedAxis

    def getTimeInfoLbl(self, timeRng):
        # Convert time ticks to tick strings
        startTime, endTime = timeRng
        startStr = self.window.getTimestampFromTick(startTime)
        endStr = self.window.getTimestampFromTick(endTime)

        # Remove day of year
        startStr = startStr[:4] + startStr[8:]
        endStr = endStr[:4] + endStr[8:]

        # Create time label widget and add to grid layout
        timeLblStr = 'Time Range: ' + startStr + ' to ' + endStr
        timeLbl = pg.LabelItem(timeLblStr)
        timeLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        return timeLbl

    def showValue(self, freq, time, prefix, lastCalc):
        # Takes x,y values and uses them to find/display the power value
        if lastCalc is None:
            return

        if self.ui.scaleModeBox.currentText() == 'Logarithmic':
            freq = 10 ** freq

        # Find grid indices corresponding to the point
        times, freqs, powerGrid = lastCalc
        numRows, numCols = powerGrid.shape
        freqIndex = max(bisect.bisect(freqs, freq), 0)
        timeIndex = max(bisect.bisect(times, time)-1, 0)

        if freq >= freqs[-1] or time <= times[0] or time >= times[-1]:
            self.ui.statusBar.clearMessage()
            return

        # Extract the grid's frequency and power values
        val = powerGrid[freqIndex][timeIndex]

        # Create and display the freq/power values in the status bar
        freqStr = NumLabel.formatVal(freq, 5)
        valStr = NumLabel.formatVal(val, 5)
        msg = prefix +'('+freqStr+', '+valStr+')'
        self.ui.statusBar.showMessage(msg)

    def openLineTool(self):
        self.closeLineTool()
        dtaRange = self.getDataRange()
        self.lineTool = SpectraLineEditor(self, self.window, dtaRange)
        self.lineTool.show()

    def closeLineTool(self):
        if self.lineTool:
            self.lineTool.close()
            self.lineTool = None

    def setSavedLine(self, lineInfo):
        self.savedLineInfo = lineInfo

    def clearSavedLine(self):
        self.savedLineInfo = None

    def addSavedLine(self):
        for lineInfo in self.savedLineInfo:
            # Get state info
            a, b = self.getDataRange()
            arbDstr = self.window.DATASTRINGS[0]
            times = self.window.getTimes(arbDstr, 0)[0][a:b]

            # Extract line info + generate the new line and add it to the plot
            expr, color, width, style = lineInfo
            lineEditor = SpectraLineEditor(self, self.window, (a,b))
            dta = lineEditor.evalExpr(expr, a, b)
            lineItem = lineEditor.createLine(dta, times, color, style, width)
            self.addLineToPlot(lineItem)

    def openMaskTool(self):
        self.closeMaskTool()
        self.maskTool = MaskTool(self, self.getToolType())
        self.maskTool.show()

    def closeMaskTool(self):
        if self.maskTool:
            self.maskTool.close()
            self.maskTool = None
    
    def getToolType(self):
        pass

    def addLineToPlot(self, line):
        pass

    def exportData(self, window, plt, fftParam):
        # Get the filename to save data to from user
        filename = window.saveFileDialog()
        if filename is None:
            return

        # Format information about the data file's and selected time range
        dataFile = 'unknown'
        if len(window.FIDs) > 0:
            names = [os.path.split(FID.name)[1] for FID in self.window.FIDs]
            dataFile = ','.join(names)
        dataFile = 'File(s): ' + dataFile + '\n'

        timeFmtStr = 'yyyy MMM dd HH:mm:ss.zzz'
        startTime = self.ui.timeEdit.start.dateTime().toString(timeFmtStr)
        endTime = self.ui.timeEdit.end.dateTime().toString(timeFmtStr)
        timeRangeStr = 'Time Range: ' + startTime + ' to ' + endTime + '\n'

        dataInfo = [dataFile, timeRangeStr]

        self.writeExportData(filename, plt, dataInfo, fftParam)

    def writeExportData(self, filename, plt, dataInfo, fftParam):
        # Get plot grid info
        freqs, times, grid, mappedColors = plt.getSavedPlotInfo()
        freqs = freqs[1:][::-1] # Reverse frequency order, skip extended values
        times = times[:-1]
        grid = grid[::-1]
        mappedColors = mappedColors[::-1]

        # String formatting lambda functions
        fmtNum = lambda n : np.format_float_positional(n, precision=7, trim='0')
        fmtStr = lambda v : '{:<15}'.format(fmtNum(v))

        # Open file for writing
        fd = open(filename, 'w')

        # Write file info + FFT parameters in first
        for line in dataInfo:
            fd.write(line)

        for lbl, val in zip(['FFT Interval: ', 'FFT Shift: ', 'Bandwidth: ', 
            'Detrended: '], fftParam):
            fd.write(lbl + str(val) + '\n')

        # Get SCET row string
        timeRowStr = ('{:<15}'.format('Freq\SCET'))
        for t in times[:-1]:
            timeRowStr += fmtStr(t)
        timeRowStr += '\n'

        # Write each row of data and the corresponding frequency
        # for both the calculated values and mapped colors
        valRowStr = ''
        colorRowStr = ''
        for i in range(0, len(freqs)):
            # Get row information
            freqStr = fmtStr(freqs[i])
            rowVals = grid[i]
            colors = mappedColors[i]

            # Add current frequency label
            valRowStr += freqStr
            colorRowStr += freqStr

            # Write down grid values
            for e in rowVals:
                valRowStr += fmtStr(e)

            # Write down cplor values in hex format
            for color in colors:
                colorRowStr += '{:<15}'.format(str(color.name()))

            valRowStr += '\n'
            colorRowStr += '\n'

        # Write strings into file
        fd.write('\nGrid Values\n')
        fd.write(timeRowStr)
        fd.write(valRowStr)
        fd.write('\n')

        fd.write('Mapped Grid Colors (Hexadecimal)\n')
        fd.write(timeRowStr)
        fd.write(colorRowStr)
        fd.close()

    def getAddBtn(self):
        ''' Returns button with arrow pointing to bottom right '''
        return AddBtn()

    def addToMain(self, sigHand=None, plt=None):
        ''' Adds spectrogram to main plot grid '''
        if plt is None:
            plt = self.plotItem

        # If there is a plot generated
        if plt:
            # Get its specData and create a copy
            specData = plt.getSpecData()
            specCopy = copy(specData)

            # Remove analysis from plot name to make it shorter
            name = specCopy.get_name()
            if 'Analysis' in name:
                name = ' '.join([elem.strip(' ') for elem in name.split('Analysis')])
            specCopy.set_name(name)

            # Add spectrogram to main plot grid and close window
            self.window.addSpectrogram(specCopy)
            self.close()

class SpectraLegend(GradLegend):
    def __init__(self, offsets=(31, 48)):
        GradLegend.__init__(self)
        topOff, botOff = offsets
        self.setOffsets(topOff, botOff)
        self.logMode = False

    def getCopy(self):
        newLegend = SpectraLegend()
        newLegend.setRange(self.getGradient(), self.getValueRange())
        return newLegend

    def setLogMode(self, logMode):
        self.logMode = logMode
        if logMode:
            self.setTickSpacing(1, 0.5)

    def logModeSetting(self):
        return self.logMode

class SpectraGridItem(pg.PlotCurveItem):
    '''
    Grid version of a set of SpectraLine items; Optimized for performance
    '''
    def __init__(self, freqs, colors, times, window=None, *args, **kargs):
        # Takes the y-values, mapped color values, and time ticks
        self.freqs = freqs
        self.times = times

        # Attributes used when drawing as an SVG image
        self.drawEdges = False # Draw corners so they are visible
        self.offset = 0 # Tick offset for x-axis

        # Flatten colors into a list so they can be zipped with the pre-generated
        # subpaths in the paint method
        self.flatColors = []
        for row in colors:
            self.flatColors.extend(row)

        # Used to update window's status bar w/ the clicked value if passed
        self.window = window
        self.prevPaths = []

        times = list(times) * len(self.freqs)
        freqs = list(freqs) * len(self.times)
        pg.PlotCurveItem.__init__(self, x=times, y=freqs, *args, **kargs)

    def getGridData(self):
        return (self.freqs, self.flatColors, self.times)

    def setEdgeMode(self, val=True):
        '''
        Enables/disables drawing edge paths
        '''
        self.drawEdges = val

    def setOffset(self, ofst=None):
        '''
        Sets the value offset for time ticks
        '''
        self.times = np.array(self.times) + self.offset
        if ofst is None:
            self.offset = 0
        else:
            self.times -= ofst
            self.offset = ofst

        times = list(self.times) * len(self.freqs)
        self.prevPaths = []
        self.setData(x=times, y=self.yData)

    def setupPath(self, p):
        # Creates subpath for each rect in the grid
        pt = QtCore.QPointF(0, 0)
        for r in range(0, len(self.freqs)-1):
            y0 = self.freqs[r]
            y1 = self.freqs[r+1]
            height = y1 - y0
            for c in range(0, len(self.times)-1):
                x0 = self.times[c]
                x1 = self.times[c+1]

                # Upper left corner
                p2 = QtGui.QPainterPath(pt)
                p2.addRect(x0, y0, x1-x0, height)
                yield p2

    def paint(self, p, opt, widget):
        if self.xData is None or len(self.xData) == 0:
            return

        p.setRenderHint(p.Antialiasing, False)

        # Generate subpaths if they haven't been generated yet
        if self.prevPaths == []:
            self.prevPaths = list(self.setupPath(p))

        # Draw edges and fill rects for every point if exporting image
        if self.drawEdges:
            for color, subpath in zip(self.flatColors, self.prevPaths):
                p.setBrush(color)
                p.setPen(pg.mkPen(color))
                p.drawPath(subpath)
            return

        # Draws filled rects for every point using designated colors
        for color, subpath in zip(self.flatColors, self.prevPaths):
            p.fillPath(subpath, color)

    def linkToStatusBar(self, window):
        self.window = window

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

class SpectraLine(pg.PlotCurveItem):
    def __init__(self, freq, colors, times, window=None, *args, **kargs):
        # Takes the y-values, mapped color values, and time ticks
        self.freq = freq
        self.colors = colors
        self.times = times
        self.drawEdges = False
        # Used to update window's status bar w/ the clicked value if passed
        self.window = window
        self.prevPaths = []

        yVals = [freq]*len(times)
        pg.PlotCurveItem.__init__(self, x=times, y=yVals, *args, **kargs)
    
    def getLineData(self):
        return (self.freq, self.times, self.colors)

    def setupPath(self, p):
        yVal = self.yData[0]
        # Draws filled rects for every point using designated colors
        for pairNum in range(0, len(self.colors)):
            # Create a rectangle path
            x0 = self.times[pairNum]
            x1 = self.times[pairNum+1]
            fillLevel = self.opts['fillLevel']
            pt1 = QtCore.QPointF(x0, yVal)
            p2 = QtGui.QPainterPath(pt1)
            p2.addRect(x0, fillLevel, x1-x0, yVal-fillLevel)
            self.prevPaths.append(p2)

    def getCornerPaths(self):
        # Generates paths for the bottom and right edges of each 'square'
        yVal = self.yData[0]
        paths = []
        viewPixelSize = self.getViewBox().viewPixelSize()
        pixWidth, pixHeight = viewPixelSize
        pixWidth = pixWidth/2
        # Draws filled rects for every point using designated colors
        for pairNum in range(0, len(self.colors)):
            # Create a rectangle path
            x0 = self.times[pairNum]
            x1 = self.times[pairNum+1]
            fillLevel = self.opts['fillLevel']
            pt1 = QtCore.QPointF(x0+pixWidth, fillLevel)
            p2 = QtGui.QPainterPath(pt1)
            p2.lineTo(x1, fillLevel)
            p2.lineTo(x1, yVal)
            paths.append(p2)
        return paths

    def paint(self, p, opt, widget):
        if self.xData is None or len(self.xData) == 0:
            return

        p.setRenderHint(p.Antialiasing, False)

        if self.prevPaths == []:
            self.setupPath(p)

        if self.drawEdges: # Draw edge paths when exporting image as SVG
            cornerPaths = self.getCornerPaths()
            for pairNum in range(0, len(self.colors)):
                color = self.colors[pairNum]
                p2 = cornerPaths[pairNum]
                pen = pg.mkPen(color)
                p.setPen(pen)
                p.drawPath(p2)

        # Draws filled rects for every point using designated colors
        for pairNum in range(0, len(self.colors)):
            # Create a rectangle path
            p2 = self.prevPaths[pairNum]
            # Find the corresponding color and fill the rectangle
            color = self.colors[pairNum]
            p.fillPath(p2, color)
    
    def linkToStatusBar(self, window):
        self.window = window

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

class RGBGradient(QtGui.QLinearGradient):
    def __init__(self):
        super().__init__()
        rgbBlue = (25, 0, 245)
        rgbBlueGreen = (0, 245, 245)
        rgbGreen = (50, 245, 0)
        rgbYellow = (245, 245, 0)
        rgbRed = (245, 0, 25)
        self.colorPos = [0, 1/3, 0.5, 2/3, 1]
        self.colors = [rgbBlue, rgbBlueGreen, rgbGreen, rgbYellow, rgbRed]

        for color, pos in zip(self.colors, self.colorPos):
            self.setColorAt(pos, QtGui.QColor(color[0], color[1], color[2]))
    
    def getColors(self):
        return self.colors
    
    def getColorPos(self):
        return self.colorPos

class PhaseGradient(QtGui.QLinearGradient):
    # Center is RGB gradient, but edges are wrapped with pink color
    def __init__(self):
        super().__init__()
        self.valueRange = (-180, 180)
        colors = RGBGradient().getColors()
        colorPos = RGBGradient().getColorPos()
        totFrac = 1 / (len(colors) + 1) # Portion of gradient that should be pink
        frac = totFrac / 2 # Portion of gradient on each *edge* that should be pink
        self.colors = [(225, 0, 255)] + colors + [(225, 0, 255)]
        self.colorPos = [0] + list(np.array(colorPos) * (1-totFrac) + frac) + [1]

        for color, pos in zip(self.colors, self.colorPos):
            self.setColorAt(pos, QtGui.QColor(color[0], color[1], color[2]))

    def getColors(self):
        return self.colors
    
    def getColorPos(self):
        return self.colorPos

class SpecData():
    def __init__(self, y_vals, x_vals, grid_vals, color_rng=None, 
        log_color=False, mask_info=None, log_y=False,
        y_label=[], legend_label=[], name=''):
        # Store plot values
        self.y_bins = y_vals
        self.x_bins = x_vals
        self.grid = grid_vals
        self.grid_range = None

        # Store plot visual attributes
        self.val_range = color_rng
        self.log_color = log_color
        self.mask_info = mask_info
        self.log_scale = log_y
        self.grad_range = None

        # Default gradient is RGB gradient
        self.gradient_stops = []
        grad = RGBGradient()
        self.set_gradient(grad)

        # Store plot label information
        self.y_label = y_label
        self.legend_label = legend_label
        self.name = name

    def get_name(self):
        return self.name

    def get_labels(self):
        return self.y_label, self.legend_label

    def get_bins(self):
        return self.y_bins, self.x_bins

    def get_grid(self):
        return self.grid

    def log_y_scale(self):
        return self.log_scale

    def log_color_scale(self):
        return self.log_color

    def get_y_label(self):
        return self.y_label

    def get_legend_label(self):
        return self.legend_label

    def get_grid_range(self):
        if self.grid_range is None:
            minVal, maxVal = np.min(self.grid), np.max(self.grid)
            self.grid_range = (minVal, maxVal)
        return self.grid_range

    def get_value_range(self):
        if self.val_range is None:
            minval, maxval = self.get_grid_range()
            if self.log_color:
                minval = np.log10(np.min(self.grid[self.grid>0]))
                maxval = np.log10(np.max(self.grid[self.grid>0]))
            return minval, maxval
        else:
            minval, maxval = self.val_range
            if self.log_color:
                minval = np.log10(minval)
                maxval = np.log10(maxval)
            return minval, maxval

    def get_gradient(self):
        gradient = QtGui.QLinearGradient()
        for pos, color in self.gradient_stops:
            gradient.setColorAt(pos, pg.mkColor(color))

        return gradient

    def set_gradient(self, grad):
        stops = list((a, b) for a,b in zip(grad.getColorPos(), grad.getColors()))
        self.gradient_stops = stops

    def get_mapped_grid(self, color_rng=None):
        grid = self.grid[:]
        if color_rng is None:
            color_rng = self.val_range[:]

        # Set value range as min/max of grid values if none is specified
        if self.val_range is None:
            minVal, maxVal = self.get_grid_range()
        else:
            minVal, maxVal = self.val_range

        # Map values and range to log base if necessary
        mask = None
        if self.log_color:
            if self.val_range is None:
                minVal = np.log10(np.min(grid[grid>0]))
                maxVal = np.log10(np.max(grid[grid>0]))
            else:
                minVal = np.log10(minVal)
                maxVal = np.log10(maxVal)
            # Mask out any values that the log can't be computed for
            mask = (grid <= 0)
            grid[mask] = 1
            grid = np.log10(grid)
        
        self.grad_range = (minVal, maxVal)

        # Create color map
        rng = maxVal - minVal
        positions, colors = [], []
        for pos, color in self.gradient_stops:
            positions.append(pos)
            colors.append(color)

        valStops = []
        for pos in positions:
            valStop = minVal + (rng * pos)
            valStops.append(valStop)
        colorMap = pg.ColorMap(valStops, colors)

        # Map values using color map and set all invalid log values to white
        mappedGrid = colorMap.map(grid)
        if mask is not None:
            mappedGrid[mask] = (255, 255, 255, 255)

        # Apply mask if not drawing outline
        if self.mask_info:
            mask, maskColor, maskOutline = self.mask_info
            r, g, b = maskColor
            if not maskOutline:
                mappedGrid[mask] = (r, g, b, 255)

        return mappedGrid

    def set_mask(self, mask_info):
        self.mask_info = mask_info

    def get_mask(self):
        return self.mask_info

    def mask_outline(self):
        if self.mask_info:
            mask, maskColor, maskOutline = self.mask_info
            r, g, b = maskColor
            pen = pg.mkPen(maskColor)

            if maskOutline:
                maskYVals = self.y_bins[:]
            else:
                return None

            if self.log_y_scale:
                maskYVals = np.log10(maskYVals)

            maskOutline = MaskOutline(mask, list(maskYVals), self.x_bins, pen=pen)
            return maskOutline

        return None

    def set_color_scale(self, val):
        self.log_color_scale = val

    def set_val_range(self, rng):
        self.val_range = rng

    def set_y_label(self, lbl):
        self.y_label = lbl

    def set_legend_label(self, lbl):
        self.legend_label = lbl

    def set_name(self, name):
        self.name = name

class SimpleColorPlot(MagPyPlotItem):
    def __init__(self, epoch, logYScale=False, vb=None, axItems=None):
        self.gradient = RGBGradient()
        self.mappedGrid = None # Mapped colors
        self.xTicks = None # m-length
        self.yTicks = None # n-length
        self.logYScale = logYScale # Y-axis scaling mode
        self.logColor = False # Color scaling mode
        self.gridItem = None # Grid PlotCurveItem
        self.valueRange = (0, 1) # Gradient legend value range
        self.baseOffset = None # Parameter used when exporting as SVG
        self.maskInfo = None

        # Initialize default viewbox and axis items
        vb = SpectrogramViewBox() if vb is None else vb

        MagPyPlotItem.__init__(self, epoch=epoch, viewBox=vb)
        if logYScale:
            self.setLogMode(x=False, y=True)

        # Additional plot adjustments to tick lengths, axis z-values, etc.
        self.plotSetup()

    def isSpecialPlot(self):
        return True

    def getPlotInfo(self):
        info = (self.mappedGrid, self.xTicks, self.yTicks, self.logYScale, 
            self.logColor, self.valueRange, self.maskInfo)
        return info
    
    def loadPlotInfo(self, plotInfo):
        grid, x, y, logY, logColor, valRng, maskInfo = plotInfo

        # Apply mask to color mapped grid
        if maskInfo:
            mask, maskColor, outline = maskInfo
            r, g, b = maskColor
            grid[mask] = (r, g, b, 255)

        self.valueRange = valRng
        self.logColor = logColor
        self.setMappedGrid(grid, y, x)
        self.fillPlot()

    def getColor(self, rgb):
        r, g, b, a = rgb
        return QtGui.QColor(r, g, b)

    def getGradLegend(self, logMode=True, offsets=None, cstmTicks=None):
        # Use default offsets if none passed
        if offsets:
            gradLegend = SpectraLegend(offsets)
        else:
            gradLegend = SpectraLegend()
        gradLegend.setRange(self.gradient, self.valueRange)

        # Set log mode
        gradLegend.setLogMode(logMode)

        return gradLegend

    def mapGrid(self, grid, valRng=None, logColorScale=False):
        # Set value range as min/max of grid values if none is specified
        if valRng is None:
            minVal, maxVal = np.min(grid), np.max(grid)
        else:
            minVal, maxVal = valRng

        # Map values and range to log base if necessary
        mask = None
        if logColorScale:
            if valRng is None:
                minVal = np.log10(np.min(grid[grid>0]))
                maxVal = np.log10(np.max(grid[grid>0]))
            else:
                minVal = np.log10(minVal)
                maxVal = np.log10(maxVal)
            # Mask out any values that the log can't be computed for
            mask = (grid <= 0)
            grid[mask] = 1
            grid = np.log10(grid)

        self.valueRange = (minVal, maxVal)

        # Create color map
        rng = maxVal - minVal
        positions = self.getGradient().getColorPos()
        colors = self.getGradient().getColors()
        valStops = []
        for pos in positions:
            valStop = minVal + (rng * pos)
            valStops.append(valStop)
        colorMap = pg.ColorMap(valStops, colors)

        # Map values using color map and set all invalid log values to white
        mappedGrid = colorMap.map(grid)
        if mask is not None:
            mappedGrid[mask] = (255, 255, 255, 255)
        
        self.logColor = logColorScale

        return mappedGrid

    def setMappedGrid(self, mapGrid, yTicks, xTicks):
        # TODO: Check that shapes of input are correct (m-1, n-1), m, n
        m = len(yTicks)
        n = len(xTicks)
        r, c = len(mapGrid), len(mapGrid[0])

        # Set attributes and return True
        self.mappedGrid = mapGrid
        self.xTicks = xTicks
        self.yTicks = yTicks
        
        return True

    def setYScaling(self, logScale=False):
        self.logYScale = logScale

    def setGradient(self, gradient):
        self.gradient = gradient
    
    def getGradient(self):
        return self.gradient

    def fillPlot(self):
        # Takes the mapped grid values (as RGB tuples) and the corresponding
        # x/y ticks and generates each line accordingly
        yTicks = self.yTicks[:]
        if self.logYScale:
            yTicks = np.log10(yTicks)

        colorGrid = [list(map(self.getColor, self.mappedGrid[i])) for i in range(0, len(yTicks)-1)]
        self.gridItem = SpectraGridItem(yTicks, colorGrid, self.xTicks)
        self.addItem(self.gridItem)

        # Set x and y ranges to min/max values of each range w/ no padding
        self.enableAutoRange(x=False, y=False)
        self.setYRange(yTicks[0], yTicks[-1], 0.0)
        self.setXRange(self.xTicks[0], self.xTicks[-1], 0.0)

    def plotSetup(self):
        # Shift axes ticks outwards instead of inwards
        la = self.getAxis('left')
        la.setStyle(tickLength=8)

        ba = self.getAxis('bottom')
        ba.setStyle(tickLength=8)
        ba.autoSIPrefix = False # Disable extra label used in tick offset

        # Hide tick marks on right/top axes
        for ax in ['right', 'top']:
            self.showAxis(ax)
            axis = self.getAxis(ax)
            axis.setStyle(showValues=False, tickLength=0)

        # Draw axes on top of any plot items (covered by SpectraLines o/w)
        for ax in ['bottom', 'right', 'top', 'left']:
            self.getAxis(ax).setZValue(1000)

        # Disable mouse panning/scaling
        self.setMouseEnabled(y=False, x=False)
        self.setDownsampling(mode='subsample')
        self.hideButtons()

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

    # Prepares plot item for export as an SVG image by adjusting large time values
    def prepareForExport(self):
        # Offset times for each data tiem
        pdis = self.listDataItems()
        self.baseOffset = self.xTicks[0]
        for pdi in pdis:
            if pdi == self.gridItem:
                pdi.setEdgeMode(True)
                pdi.setOffset(self.baseOffset)
            else:
                pdi.setData(x=pdi.xData - self.baseOffset, y=pdi.yData)
            pdi.update()

        # Adjust time tick offset
        ba = self.getAxis('bottom')
        ba.tickOffset = self.baseOffset

        # Offset viewbox range/state as well
        vb = self.getViewBox()
        xRange, yRange = vb.viewRange()
        xMin, xMax = xRange
        vb.setRange(xRange=(xMin-self.baseOffset, xMax-self.baseOffset), padding=0)
        vb.prepareForPaint()
        vb.update()

    def resetAfterExport(self):
        # Add in offset back into all spectra lines
        pdis = self.listDataItems()
        if self.baseOffset:
            for pdi in pdis:
                if pdi == self.gridItem:
                    pdi.setEdgeMode(False)
                    pdi.setOffset(None)
                else:
                    pdi.setData(x=pdi.xData+self.baseOffset, y=pdi.yData)
                pdi.path = None
        else:
            return

        # Reset tick offset
        ba = self.getAxis('bottom')
        ba.tickOffset = 0

        # Add offset back into view range and reset viewbox state
        vb = self.getViewBox()
        xRange, yRange = vb.viewRange()
        xMin, xMax = xRange
        vb.setRange(xRange=(xMin+self.baseOffset, xMax+self.baseOffset), padding=0)
        vb.prepareForPaint()
        vb.update()

        self.baseOffset = None

class SpectrogramViewBox(pg.ViewBox):
    # Optimized viewbox class, removed some steps in addItem/clear methods
    def __init__(self, *args, **kargs):
        self.lastScene = None
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

class SpectrogramPlotItem(SimpleColorPlot):
    def __init__(self, epoch, logMode=False, *args, **kwargs):
        super().__init__(epoch, logMode, *args, **kwargs)
        self.savedPlot = None
        self.specData = None

        # Initialize default pg.PlotItem settings
        self.plotAppr = None
        self.plotApprAct = None
        self.plotMenuEnabled = True
        self.exportAct = None
        self.exportEnabled = False

    def setPlotMenuEnabled(self, val=True):
        if val:
            self.plotMenuEnabled = True
        else:
            self.plotMenuEnabled = False

    def openPlotAppearance(self):
        self.closePlotAppearance()
        self.plotAppr = DynamicPlotApp(self, [self])
        self.plotAppr.show()

    def closePlotAppearance(self):
        if self.plotAppr:
            self.plotAppr.close()
            self.plotAppr = None

    def setExportEnabled(self, linkFunc):
        self.exportEnabled = True
        self.exportAct = QtWidgets.QAction('Save Plot Data...')
        self.exportAct.triggered.connect(linkFunc)

    def getPlotApprMenu(self):
        self.plotApprAct = QtWidgets.QAction('Change Plot Appearance...')
        self.plotApprAct.triggered.connect(self.openPlotAppearance)
        self.stateGroup.autoAdd(self.plotApprAct)
        return self.plotApprAct

    def getContextMenus(self, event):
        if self.plotMenuEnabled or self.exportEnabled:
            actList = [self.ctrlMenu]
            if self.exportEnabled:
                self.stateGroup.autoAdd(self.exportAct)
                actList = [self.exportAct] + actList
            if self.plotMenuEnabled:
                plotApp = self.getPlotApprMenu()
                actList = [plotApp] + actList
            return actList
        else:
            return self.ctrlMenu

    def loadPlot(self, specData, winFrame=None):
        ''' Loads spectrogram from specData object '''
        # Extract grid info
        y, x = specData.get_bins()
        grid = specData.get_grid()

        # Get color and mask info
        color_rng = specData.val_range
        mask_info = specData.get_mask()

        # Set log y scaling
        self.logYScale = specData.log_y_scale()
        if self.logYScale:
            self.setLogMode(x=False, y=True)
        
        # Set gradient
        gradient = specData.get_gradient()
        self.setGradient(gradient)

        # Generate plot
        self.createPlot(y, grid, x, color_rng, specData.log_color_scale(),
            winFrame, mask_info)

    # Takes the y-vals (length m), time ticks (length n), a matrix of values 
    # (of shape (m-1) x (n-1)), and a tuple of min/max values repres. by color gradient
    def createPlot(self, yVals, valueGrid, timeVals, colorRng, logColorScale=True, 
        winFrame=None, maskInfo=None):
        # Map values in grid to RGB colors
        self.specData = SpecData(yVals, timeVals, valueGrid, colorRng, logColorScale,
            maskInfo, self.logYScale)
        mappedGrid = self.specData.get_mapped_grid()
        self.valueRange = self.specData.grad_range
        self.logColor = self.specData.log_color_scale()

        # Set the mapped grid colors for this plot and generate
        self.setMappedGrid(mappedGrid, yVals, timeVals)
        self.fillPlot()

        # Set the time axis label
        timeLbl = self.getAxis('bottom').tm.getTimeLabel(timeVals[-1]-timeVals[0])
        self.getAxis('bottom').setLabel(timeLbl)

        # Add in mask outlines
        mask_outline = self.specData.mask_outline()
        if mask_outline is not None:
            self.addItem(mask_outline)

        # Link tool's statusBar to clicks on the plot
        if winFrame:
            self.linkToStatusBar(winFrame)

    def getSpecData(self):
        return self.specData

    def savePlotInfo(self, freqs, times, grid):
        self.savedPlot = (freqs, times, grid)

    def getSavedPlotInfo(self):
        # Build plot info from specData object
        freqs, times = self.specData.get_bins()
        grid = self.specData.get_grid()
        mappedGrid = self.specData.get_mapped_grid()
        mappedColors = [list(map(self.getColor, row)) for row in mappedGrid]
    
        return (freqs, times, grid, mappedColors)

    def linkToStatusBar(self, window):
        # Link clicks on this plot to the window's statusBar to display
        # relevant values
        self.gridItem.linkToStatusBar(window)

class PhaseSpectrogram(SpectrogramPlotItem):
    def __init__(self, epoch, logMode=True):
        super().__init__(epoch, logMode)
        self.valueRange = (-180, 180)
        self.gradient = PhaseGradient()

class MaskOutline(pg.PlotCurveItem):
    def __init__(self, mask, yVals, times, *args, **kargs):
        self.path = None

        # Store mask grid, freqs, times
        self.mask = mask
        self.freqs = yVals
        self.times = times

        # Map values used to indicate whether to draw a specific edge
        # for a given 'rect' at a given index
        self.leftBase = 1
        self.topBase = 2
        self.rightBase = 4
        self.bottomBase = 8

        # Plot values passed to PlotCurveItem
        xVals = list(times) * len(yVals)
        yVals = list(yVals) * len(times)

        pg.PlotCurveItem.__init__(self, x=xVals, y=yVals, *args, **kargs)

    def shiftMask(self, mask, direction):
        rows, cols = mask.shape
        # Make a copy of the mask and shift it by a single row/column in 
        # the given direction
        maskCopy = np.empty(mask.shape, dtype=np.int32)
        if direction == 'Up':
            # Drop first row and fill last row w/ copy of mask's last row
            maskCopy[:-1] = mask[1:]
            maskCopy[-1] = mask[-1]
        elif direction == 'Down':
            # Drop last row and fill first row w/ copy of mask's first row
            maskCopy[1:] = mask[:-1]
            maskCopy[0] = mask[0]
        elif direction == 'Left':
            # Drop first column and fill last col w/ copy of mask's last col
            maskCopy[:,0:-1] = mask[:,1:]
            maskCopy[:,-1] = mask[:,-1]
        else:
            # Drop last column and fill first col w/ copy of mask's first col
            maskCopy[:,1:] = mask[:,0:-1]
            maskCopy[:,0] = mask[:,0]
        
        return maskCopy

    def getMaskOutline(self, mask):
        shiftBases = [self.leftBase, self.rightBase, self.topBase, self.bottomBase]
        shiftDirs = ['Right', 'Left', 'Up', 'Down']

        # Create a grid of zeros (zero = no edge is drawn for given point)
        maskOutlineVals = np.zeros(mask.shape, dtype=np.int32)
        # For each edge direction
        for base, shiftDir in zip(shiftBases, shiftDirs):
            # Shift the mask in the given direction and do an XOR w/ original mask
            # to see if there is a sign change
            shiftedMask = self.shiftMask(mask, shiftDir)
            maskAndOp = np.logical_xor(mask, shiftedMask)

            # Do an additional AND op w/ the original mask to get only points
            # that are part of the masked values (e.g. unmasked value that was masked)
            maskAndOp = np.logical_and(mask, maskAndOp)

            # Multiply mask by given base value and add it to grid of values
            baseMask = np.array(maskAndOp, dtype=np.int32) * base
            maskOutlineVals = maskOutlineVals + baseMask

        return maskOutlineVals
    
    def getPoint(self, x, y):
        return QtCore.QPointF(x, y)

    def setupPath(self):
        # Get values indicating which edges to draw at each point (sum of powers of 2)
        maskOutline = self.getMaskOutline(self.mask)
        path = QtGui.QPainterPath(QtCore.QPointF(0, 0))

        # Draws filled rects for every point using designated colors
        for i in range(0, len(self.freqs)-1):
            y0 = self.freqs[i]
            y1 = self.freqs[i+1]
            for j in range(0, len(self.times)-1):
                # Skip values w/ no edges to draw
                boxVal = maskOutline[i][j]
                if boxVal <= 0:
                    continue

                x0 = self.times[j]
                x1 = self.times[j+1]

                # Move path to start (bottom left corner) and initialize the
                # points at each corner in a clockwise rotation
                path.moveTo(self.getPoint(x0, y0))
                points = [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]
                points = [self.getPoint(x, y) for x, y in points]
                bases = [self.leftBase, self.topBase, self.rightBase, self.bottomBase]

                # Do a bitwise AND operation to see if we need to draw a given
                # edge to the next corner; Otherwise, just move the path position
                # to that corner
                for pt, base in zip(points, bases):
                    if np.bitwise_and(boxVal, base):
                        path.lineTo(pt)
                    else:
                        path.moveTo(pt)

        self.path = path
        return path

    def paint(self, p, opt, widget):
        # Setup path on first paintEvent
        if self.path is None:
            path = self.setupPath()
        else:
            path = self.path

        # Set antialiasing parameter for pen to False
        if self._exportOpts is not False:
            aa = self._exportOpts.get('antialias', False)
        else:
            aa = self.opts['antialias']
        p.setRenderHint(p.Antialiasing, False)

        # Set pen width to 2
        cp = pg.mkPen(self.opts['pen'])
        cp.setWidth(2)
        cp.setJoinStyle(QtCore.Qt.MiterJoin)
        p.setPen(cp)

        # Draw path edges using given pen
        p.strokePath(path, cp)

from .maskToolBase import MaskTool
