from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from MagPy4UI import TimeEdit, NumLabel
from pyqtgraphExtensions import GridGraphicsLayout, LogAxis, MagPyAxisItem, DateAxis, StackedAxisLabel
import bisect
import functools
from mth import Mth
from layoutTools import BaseLayout
from simpleCalculations import ExpressionEvaluator

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

        # Set up clear/plot buttons and status bar
        self.fixLine = QtWidgets.QCheckBox('Fix Line')
        self.fixLine.setToolTip('Replot line after plot is updated')
        self.clearBtn = QtWidgets.QPushButton('Clear')
        self.addBtn = QtWidgets.QPushButton('Plot')
        self.statusBar = QtWidgets.QStatusBar()

        layout.addWidget(self.statusBar, 3, 0, 1, 1)
        layout.addWidget(self.fixLine, 3, 1, 1, 1)
        layout.addWidget(self.clearBtn, 3, 2, 1, 1)
        layout.addWidget(self.addBtn, 3, 3, 1, 1)

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
        self.dataRange = dataRange
        self.plottedLine = None
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
            expEval = ExpressionEvaluator(exprStr, self.window, self.dataRange)
            name, exprLst = expEval.splitString()
            stack = expEval.createStack(exprLst)
            res = stack.evaluate()
            dta = res.evaluate()

            # Reshape constants to match times length
            if res.isNum():
                dta = [dta] * (eI-sI)

            # Adjust values if plot is in log scale
            if self.isLogMode():
                dta = np.log10(dta)

            return dta

        except:
            return None

    def addToPlot(self):
        self.clearPlot()
        sI, eI = self.dataRange
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
        self.saveLineInfo()

    def clearPlot(self):
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

    def saveLineInfo(self):
        # Extract parameters used to plot current line
        expr = self.ui.textBox.toPlainText()
        color = self.ui.lineColor
        width = self.ui.lineWidth.value()
        style = self.ui.lineStyles[self.ui.lineStyle.currentText()]

        self.plottedLine = (expr, color, width, style)

        if self.ui.fixLine.isChecked(): # Save in outer frame if fixLine box is checked
            self.fixedLineToggled(True)

    def clearLineInfo(self):
        self.plottedLine = None

    def fixedLineToggled(self, val):
        if val:
            self.spectraFrame.setSavedLine(self.plottedLine)
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

class ColorBar(pg.GraphicsLayout):
    def __init__(self, gradient, parent=None):
        super().__init__(parent=None)
        self.gradient = gradient
        self.setMaximumWidth(50)
        self.setMinimumWidth(30)

    def setGradient(self, gradient):
        self.gradient = gradient

    def getGradient(self):
        return self.gradient

    def paint(self, p, opt, widget):
        ''' Fill the bounding rect w/ current gradient '''
        rect = self.boundingRect()
        self.gradient.setStart(0, rect.bottom())
        self.gradient.setFinalStop(0, rect.top())
        p.setPen(pg.mkPen((0, 0, 0)))
        p.setBrush(self.gradient)
        p.drawRect(rect)
        pg.GraphicsWidget.paint(self, p, opt,widget)

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
        self.savedLineInfo = None
        self.lineHistory = set()
        self.fftBins = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.fftBinBound = 131072

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
        if self.savedLineInfo:
            # Get state info
            a, b = self.getDataRange()
            arbDstr = self.window.DATASTRINGS[0]
            times = self.window.getTimes(arbDstr, 0)[0][a:b]

            # Extract line info + generate the new line and add it to the plot
            expr, color, width, style = self.savedLineInfo
            lineEditor = SpectraLineEditor(self, self.window, (a,b))
            dta = lineEditor.evalExpr(expr, a, b)
            lineItem = lineEditor.createLine(dta, times, color, style, width)
            self.addLineToPlot(lineItem)
        return None

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

class SpectraLine(pg.PlotCurveItem):
    def __init__(self, freq, colors, times, window=None, *args, **kargs):
        # Takes the y-values, mapped color values, and time ticks
        self.freq = freq
        self.colors = colors
        self.times = times
        # Used to update window's status bar w/ the clicked value if passed
        self.window = window
        self.prevPaths = []

        yVals = [freq]*len(times)
        pg.PlotCurveItem.__init__(self, x=times, y=yVals, *args, **kargs)

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

    def paint(self, p, opt, widget):
        if self.xData is None or len(self.xData) == 0:
            return

        p.setRenderHint(p.Antialiasing, False)

        if self.prevPaths == []:
            self.setupPath(p)

        # Draws filled rects for every point using designated colors
        for pairNum in range(0, len(self.colors)):
            # Create a rectangle path
            p2 = self.prevPaths[pairNum]
            # Find the corresponding color and fill the rectangle
            color = self.colors[pairNum]
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

class SpectrogramPlotItem(pg.PlotItem):
    def __init__(self, epoch, logMode=False):
        super(SpectrogramPlotItem, self).__init__(parent=None)
        self.logMode = logMode # Log scaling for y-axis parameter (Boolean)
        self.baseOffset = None
        self.lines = []

        # Initialize colors for color map
        rgbBlue = (25, 0, 245)
        rgbBlueGreen = (0, 245, 245)
        rgbGreen = (127, 245, 0)
        rgbYellow = (245, 245, 0)
        rgbRed = (245, 0, 25)

        self.colorPos = [0, 1/3, 0.5, 2/3, 1]
        self.colors = [rgbBlue, rgbBlueGreen, rgbGreen, rgbYellow, rgbRed]
        self.valueRange = (0, 1)

        # Create viewbox and set up custom axis items
        vb = SpectrogramViewBox()
        dateAxis = DateAxis(epoch, orientation='bottom')
        if self.logMode:
            leftAxis = LogAxis(orientation='left')
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

    def isSpecialPlot(self):
        return True

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

        self.valueRange = (minLog, maxLog)

        # Determine the non-log values the color map will use
        midPoint = (minLog + maxLog) / 2
        oneThird = (maxLog - minLog) / 3
        logLevels = [minLog, minLog+oneThird, midPoint, maxLog-oneThird, maxLog]

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
        # Use default offsets if none passed
        if offsets:
            gradLegend = SpectraLegend(offsets)
        else:
            gradLegend = SpectraLegend()

        # Create color gradient legend based on color map
        gradient = QtGui.QLinearGradient()
        for pos, rgb in zip(self.colorPos, self.colors):
            gradient.setColorAt(pos, pg.mkColor(rgb))
        gradLegend.setRange(gradient, self.valueRange)

        # Set log mode
        gradLegend.setLogMode(logMode)

        return gradLegend

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
        lowerBnd = lastVal

        stepSize = (statusEnd-statusStrt) / (len(yVals) - 1)
        currentStep = statusStrt

        self.lines = []
        # Creates a SpectraLine object for every row in value grid
        for rowIndex in range(0, len(yVals)-1):
            yVal = yVals[rowIndex+1]
            colors = list(map(self.mkRGBColor, mappedGrid[rowIndex,:]))
            pdi = self.getSpectraLine(yVal, colors, timeVals, winFrame, lastVal)
            self.addItem(pdi)
            self.lines.append(pdi)

            if winFrame: # If winFrame is passed, update progress in status bar
                currentStep += stepSize
                winFrame.ui.statusBar.showMessage('Generating plot...' + str(int(currentStep)) + '%')

            lastVal = yVal

        # Set axis ranges and update time label
        self.setXRange(timeVals[0], timeVals[-1], 0)
        self.setYRange(lowerBnd, yVals[-1], 0)
        timeLbl = self.getAxis('bottom').tm.getTimeLabel(timeVals[-1]-timeVals[0])
        self.getAxis('bottom').setLabel(timeLbl)

    # Prepares plot item for export as an SVG image by adjusting large time values
    def prepareForExport(self):
        # Offset times for each data tiem
        pdis = self.listDataItems()
        self.baseOffset = self.lines[0].times[0]
        for pdi in pdis:
            if pdi in self.lines:
                pdi.times = pdi.times - self.baseOffset
                pdi.prevPaths = []
            pdi.setData(x=pdi.xData-self.baseOffset, y=pdi.yData)
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
                if pdi in self.lines:
                    pdi.times = pdi.times + self.baseOffset
                pdi.setData(x=pdi.xData+self.baseOffset, y=pdi.yData)
                pdi.prevPaths = []
                pdi.path = None

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

class PhaseSpectrogram(SpectrogramPlotItem):
    def __init__(self, epoch, logMode=True):
        super(SpectrogramPlotItem, self).__init__(parent=None)
        SpectrogramPlotItem.__init__(self, epoch, logMode)
        self.valueRange = (-180, 180)
        self.colors = [(225, 0, 255)] + self.colors + [(225, 0, 255)]
        self.colorPos = [0]
        centerTotal = 5/6
        startStop = 1/12
        for pos in [0, 1/3, 1/2, 2/3, 1]:
            currStop = startStop + (centerTotal*pos)
            self.colorPos.append(currStop)
        self.colorPos.append(1)

    def mapValueToColor(self, valueGrid, minPower, maxPower, logColorScale):
        minVal, maxVal = self.valueRange
        stepSize = 360 / 6
        colorVals = [(minVal + (360 * pos)) for pos in self.colorPos]
        colorMap = pg.ColorMap(colorVals, self.colors)
        mappedGrid = colorMap.map(valueGrid)
        return mappedGrid

class DynamicSpectraUI(BaseLayout):
    def setupUI(self, Frame, window):
        self.Frame = Frame
        maxSizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        Frame.setWindowTitle('Dynamic Spectrogram')
        Frame.resize(1100, 900)
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

        spacer = QtWidgets.QSpacerItem(10, 1)

        self.updateBtn = QtWidgets.QPushButton('Update')
        self.updateBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.addLineBtn = QtWidgets.QPushButton('Add Line')
        self.addLineBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

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
        settingsLt.addWidget(self.addLineBtn, 1, 7, 1, 1)

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
        self.initPlot(window)
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

    def initPlot(self, window):
        # Clear previous plot
        self.glw.clear()

        # Set title and lower downsampling
        dstrTitle = self.dstrBox.currentText()

        logMode = self.scaleModeBox.currentText() == 'Logarithmic'
        self.plotItem = SpectrogramPlotItem(window.epoch, logMode=logMode)
        self.plotItem.setTitle('Dynamic Spectra Analysis'+' ('+dstrTitle+')', size='13pt')
        self.plotItem.setDownsampling(mode='subsample')

        self.glw.addItem(self.plotItem, 0, 0, 5, 4)

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
        self.glw.addItem(self.timeLbl, 5, 0, 1, 1)

    def setupGradient(self):
        # Create gradient legend and add it to the graphics layout
        gradLegend = self.plotItem.getGradLegend()
        gradLegend.setBarWidth(40)
        self.glw.addItem(gradLegend, 0, 5, 5, 1)

        # Add in legend labels and center them
        lbl = StackedAxisLabel(['Log Power', '(nT^2/Hz)'], angle=0)
        lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        lbl.layout.setContentsMargins(0,0,0,0)
        self.glw.addItem(lbl, 0, 6, 5, 1)

    def powerRngSelectToggled(self, val):
        self.powerMax.setEnabled(val)
        self.powerMin.setEnabled(val)
        self.minLbl.setEnabled(val)
        self.maxLbl.setEnabled(val)

class DynamicSpectra(QtGui.QFrame, DynamicSpectraUI, DynamicAnalysisTool):
    def __init__(self, window, parent=None):
        super(DynamicSpectra, self).__init__(parent)
        SpectraBase.__init__(self)
        DynamicAnalysisTool.__init__(self)

        # Set up sum of powers plot titles and simple keywords that they map
        # to elsewhere in the DynamicSpectra class
        self.sumPowersPlotTypes = ['|Px + Py + Pz - Pt|', 'Px + Py + Pz', 'Pt']
        stateKws = ['AbsSumPowers', 'SumPowers', 'MagPower']
        self.spStateKws = {}
        for kw, stateKw in zip(self.sumPowersPlotTypes, stateKws):
            self.spStateKws[kw] = stateKw
        self.bMagDta = [] # Stores any pre-computed b_mag data
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
        self.ui.timeEdit.start.dateTimeChanged.connect(self.updateParameters)
        self.ui.timeEdit.end.dateTimeChanged.connect(self.updateParameters)

    def closeEvent(self, ev):
        self.closeLineTool()
        self.window.endGeneralSelect()
        self.window.clearStatusMsg()
        self.wasClosed = True

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
        self.ui.initPlot(self.window)

        # Generate plot grid and spectrogram from this
        self.calculate(dataRng, interval, shift, bw, dstr)

        if self.savedLineInfo:
            self.addSavedLine()

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

    def calcSumOfPowers(self, vecDstrs, bw, startIndex, endIndex):
        # Calculates the spectra for each variable separately and returns the sum
        powers = []
        for dstr in vecDstrs:
            freqs, power = self.calcSpectra(dstr, bw, startIndex, endIndex)
            powers.append(np.array(power))

        sumOfPowers = powers[0] + powers[1] + powers[2]
        return freqs, sumOfPowers

    def calcMag(self, vecDstrs, startIndex, endIndex):
        a, b = startIndex, endIndex
        # Computes the magnitude of the vector over the selected interval
        en = self.window.currentEdit
        datas = [self.window.getData(dstr, en)[a:b] ** 2 for dstr in vecDstrs]
        mag = np.sqrt(datas[0] + datas[1] + datas[2])
        # Fill start so calculations aren't affected by offset
        mag = np.concatenate([np.empty(startIndex), mag])
        return mag

    def calcMagPower(self, bw, start, end):
        # Similar to calcSpectra but using pre-computed magnitude data
        en = self.window.currentEdit
        N = abs(end-start)
        data = self.bMagDta[start:end]
        fft = fftpack.rfft(data.tolist())
        power = self.calculatePower(bw, fft, N)
        freqs = self.calculateFreqList(bw, N)
        return freqs, power

    def calculate(self, dataRng, interval, shift, bw, dstr):
        shiftAmnt = shift
        minIndex, maxIndex = dataRng
        startIndex, endIndex = minIndex, minIndex + interval

        self.ui.statusBar.showMessage('Calculating...')

        # Check if this is a special sum-of-powers plot
        spPlot = self.spStateKws[dstr] if dstr in self.spStateKws else None

        # Get vector var names and computer magnitude of vector if necessary
        if spPlot is not None:
            vecDstrs = self.getVecDstrs(dstr)
            if spPlot == 'MagPower' or spPlot == 'AbsSumPowers':
                a, b = minIndex, maxIndex
                self.bMagDta = self.calcMag(vecDstrs, a, b)
            dstr = vecDstrs[0] # Adjusted for future call to getTimes()
        else:
            self.bMagDta = [] # Clear unused magnitude calculation

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
                    freqs, powers = self.calcSumOfPowers(vecDstrs, bw, startIndex, endIndex)
                elif spPlot == 'MagPower':
                    freqs, powers = self.calcMagPower(bw, startIndex, endIndex)
                else:
                    freqs, sumPowers = self.calcSumOfPowers(vecDstrs, bw, startIndex, endIndex)
                    freqs, magPower = self.calcMagPower(bw, startIndex, endIndex)
                    powers = np.abs(sumPowers - magPower)
            else:
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
        self.ui.addTimeInfo(xRange, self.window)

        self.ui.setupGradient()

    def showPointValue(self, freq, time):
        self.showValue(freq, time, 'Freq, Power: ', self.lastCalc)

    def addLineToPlot(self, line):
        self.ui.plotItem.addItem(line)
        self.lineHistory.add(line)

    def removeLinesFromPlot(self):
        histCopy = self.lineHistory.copy()
        for line in histCopy:
            if line in self.ui.plotItem.listDataItems():
                self.ui.plotItem.removeItem(line)
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
        self.fftDataPts = QtWidgets.QLabel()
        ptsTip = 'Total number of data points within selected time range'
        self.addPair(layout, 'Num Points: ', self.fftDataPts, 2, 3, 1, 1, ptsTip)

        self.bwBox.setValue(3)
        self.bwBox.setSingleStep(2)
        self.bwBox.setMinimum(1)

        self.addLineBtn = QtWidgets.QPushButton('Add Line')
        self.addLineBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.updtBtn = QtWidgets.QPushButton(' Update ')
        self.updtBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Column of spacers before update button
        for row in range(0, 3):
            spacer = self.getSpacer(10)
            layout.addItem(spacer, row, 6, 1, 1)

        layout.addWidget(self.updtBtn, 1, 7, 1, 1)
        layout.addWidget(self.addLineBtn, 0, 7, 1, 1)
        return layout

class DynamicCohPha(QtGui.QFrame, DynamicCohPhaUI, DynamicAnalysisTool):
    def __init__(self, window, parent=None):
        super(DynamicCohPha, self).__init__(parent)
        SpectraBase.__init__(self)
        DynamicAnalysisTool.__init__(self)
        self.ui = DynamicCohPhaUI()
        self.window = window
        self.wasClosed = False

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
        self.ui.timeEdit.start.dateTimeChanged.connect(self.updateParameters)
        self.ui.timeEdit.end.dateTimeChanged.connect(self.updateParameters)

    def closeEvent(self, ev):
        self.closeLineTool()
        self.window.endGeneralSelect()
        self.window.clearStatusMsg()
        self.wasClosed = True
        self.close()

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

        if self.checkParameters(interval, shiftAmnt, bw, nPoints) == False:
            return

        self.calculate(varA, varB, bw, logMode, indexRng, shiftAmnt, interval)

        if self.savedLineInfo: # Add any saved lines
            self.addSavedLine()

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
        self.cohPlt, self.phaPlt = self.createPlots(freqs, timeSeries, cohGrid, 
            phaGrid, logMode)

        # Adjust plot ranges and y axis labels
        timeRng = (timeSeries[0], timeSeries[-1])
        freqRng = (freqs[0], freqs[-1])
        self.adjustPlots(self.cohPlt, self.phaPlt, varA, varB, logMode, timeRng, 
            freqRng)

        # Add in time label info at bottom
        for grid in [self.ui.cohGrid, self.ui.phaGrid]:
            timeLbl = self.getTimeRangeLbl(timeSeries[0], timeSeries[-1])
            grid.addItem(timeLbl, 1, 0, 1, 1)
        self.ui.statusBar.clearMessage()

    def createPlots(self, freqs, times, cohGrid, phaGrid, logMode):
        # Generate the color mapped plots from the value grids
        cohPlt = SpectrogramPlotItem(self.window.epoch, logMode)
        phaPlt = PhaseSpectrogram(self.window.epoch, logMode)
        cohRng = (0, 1.0)
        phaRng = (-180, 180)
        cohPlt.createPlot(freqs, cohGrid, times, cohRng, winFrame=self, 
            logColorScale=False, statusStrt=0, statusEnd=50)
        phaPlt.createPlot(freqs, phaGrid, times, phaRng, winFrame=self, 
            logColorScale=False, statusStrt=50, statusEnd=100)

        # Get color gradients
        cohGrad = cohPlt.getGradLegend(logMode=False)
        cohGrad.setBarWidth(40)

        phaGrad = phaPlt.getGradLegend(logMode=False)
        phaGrad.setBarWidth(40)

        # Get color bar labels
        cohLbl = pg.LabelItem('Coherence')
        cohLbl.setFixedWidth(65)

        phaLbl = StackedAxisLabel(['Phase', '[Degrees]'], angle=0)
        phaLbl.setFixedWidth(70)

        cohGrad.setTickSpacing(0.2, 0.1)
        phaGrad.setTickSpacing(60, 30)

        # Add items into grids
        for grid, plt, grad, lbl in zip([self.ui.cohGrid, self.ui.phaGrid],
            [cohPlt, phaPlt], [cohGrad, phaGrad], [cohLbl, phaLbl]):
            grid.clear()
            grid.addItem(plt)
            grid.nextCol()
            grid.addItem(grad)
            grid.nextCol()
            grid.addItem(lbl)

        return cohPlt, phaPlt

    def adjustPlots(self, cohPlt, phaPlt, varA, varB, logMode, xRng, yRng):
        # Set titles
        subTitle = '(' + varA + ' by ' + varB + ')'
        cohPlt.setTitle('Dynamic Coherence Analysis ' + subTitle, size='13pt')
        phaPlt.setTitle('Dynamic Phase Analysis ' + subTitle, size='13pt')

        # Set time labels
        if logMode:
            a = np.log10(yRng[0])
            b = np.log10(yRng[1])
            yRng = (a, b)

        # Update plot ranges and set axis labels
        for plt in [cohPlt, phaPlt]:
            if logMode:
                plt.getAxis('left').setLabel('Log Frequency (Hz)')
            else:
                plt.getAxis('left').setLabel('Frequency (Hz)')

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