from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from copy import copy

import pyqtgraph as pg
import numpy as np
import bisect
import functools
from .layoutTools import BaseLayout
import os
from .data_util import find_gaps
from multiprocessing import Pool

class NumLabel(QtWidgets.QLabel):
    def __init__(self, val=None, prec=None):
        super(NumLabel, self).__init__(None)
        self.prec = prec
        if val is not None:
            self.setText(val)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

    def setText(self, val):
        txt = str(val)
        if self.prec is not None and not np.isnan(val):
            txt = NumLabel.formatVal(val, self.prec)
        QtWidgets.QLabel.setText(self, txt)

    def formatVal(val, prec):
        if abs(val) < 1/1000 or abs(val) > (10 ** (prec + 1)):
            txt = np.format_float_scientific(val, precision=prec)
        else:
            txt = str(np.round(val, decimals=prec))
        return txt

class ParallelGrid():
    ''' Helper functions for parallelizing dynamic
        spectra and wave analysis computations
    '''
    def group_info(n, threads):
        ''' Returns the number of groups and each group
            size based on the number of items in the
            group and the number of threads
        '''
        n = float(n)
        num_groups = max(int(np.ceil(n/threads)), 1)
        group_size = int(np.ceil(n/num_groups))

        return (num_groups, group_size)
    
    def create_groups(data_segs, threads):
        ''' Splits data segments into groups based on number
            of threads, skipping any empty lists
            and returns groups of segments and their
            orders
        '''
        n = len(data_segs)
        n_grps, grp_size = ParallelGrid.group_info(n, threads)
        
        indices = np.arange(0, n+grp_size, grp_size)
        groups = [data_segs[i:i+grp_size] for i in indices[:-1]]
        groups = [grp for grp in groups if len(grp) > 0]
        
        return groups

    def sort_groups(groups):
        indices = [group[1] for group in groups]
        sortorder = np.argsort(indices)
        sorted_grps = [groups[i][0] for i in sortorder]
        return sorted_grps

    def map_wrapper(func_info, group, index):
        (func, func_args, func_kwargs) = func_info
        results = []
        for seg in group:
            result = func(seg, *func_args, **func_kwargs)
            results.append(result)
        return (results, index)

    def parallelize(groups, func, func_args=[], func_kwargs={}):
        ''' Process groups with given function, function  args,
            and func_kwargs using a multiprocessed pool
            and return the results in sorted order
        '''
        # Create list of indices to keep track of groups ordering
        n = len(groups)
        order = list(range(n))

        # Set up arguments to pass to wrapper function
        func_infos = [(func, func_args, func_kwargs) for i in range(n)]
        map_args = list(map(list, zip(func_infos, groups, order)))

        # Create a pool and map using the map_wrapper function
        pool = Pool(len(groups))
        wrap_func = ParallelGrid.map_wrapper
        result = pool.starmap(wrap_func, map_args)

        # Sort results and close pool
        pool.close()
        result = ParallelGrid.sort_groups(result)

        return result

class GradEditor(QtWidgets.QFrame):
    def __init__(self, grad):
        self.grad = grad
        self.spec = grad.getPlot().get_specs()[0].spec
        QtWidgets.QFrame.__init__(self)
        self.setWindowTitle('Spectrogram Editor')
        self.resize(250, 200)

        self.setupUI()
        self.applyBtn.clicked.connect(self.applyChanges)

    def setupUI(self):
        layout = QtWidgets.QGridLayout(self)

        # Set up value range layout
        rangeFrm = self.setupRangeLt()

        # Set up log color scaling checkbox
        self.logColor = QtWidgets.QCheckBox('Log Color Scaling')
        self.logColor.setChecked(self.spec.log_color_scale())
        self.logColor.toggled.connect(self.scalingUpdated)

        # Set up log axis scaling checkbox
        self.logY = QtWidgets.QCheckBox('Log Y-Axis Scaling')
        self.logY.setChecked(self.spec.log_y_scale())

        self.applyBtn = QtWidgets.QPushButton('Apply')
        layout.addWidget(rangeFrm, 0, 0, 1, 1)
        layout.addWidget(self.logColor, 1, 0, 1, 1)
        layout.addWidget(self.logY, 2, 0, 1, 1)
        layout.addWidget(self.applyBtn, 3, 0, 1, 1)

    def applyChanges(self):
        # Get new value ranges
        minVal = self.minBox.value()
        maxVal = self.maxBox.value()

        # Adjust value range if on log scale
        if self.minBox.prefix() != '':
            minVal = 10 ** minVal
            maxVal = 10 ** maxVal

        valRange = (minVal, maxVal)

        # If default box is checked, use None
        if self.defaultRngBox.isChecked():
            valRange = None

        # Update color scaling
        self.spec.set_color_scale(self.logColor.isChecked())

        # Set range for specData and gradient bar
        self.spec.set_val_range(valRange)
        gradRange = self.spec.get_value_range()
        self.grad.setRange(self.grad.getGradient(), gradRange)

        # Update gradient color scale mode
        self.grad.setLogMode(self.logColor.isChecked())

        # Update plot scaling and left axis label accordingly
        log_scale = self.logY.isChecked()
        self.spec.set_y_log_scale(log_scale)

        # Update prepended 'log' label on plot axis
        plt = self.grad.getPlot()
        label = self.spec.get_y_label()
        new_label = self.update_log_label(label, log_scale)
        plt.getAxis('left').setLabel(new_label)
        self.spec.set_y_label(new_label)

        # Reload plot
        plt.clear_specs()
        plt.load_color_plot(self.spec)

    def update_log_label(self, label, log_scale):
        items = label.split(' ')
        new_label = label
        if len(items) > 0:
            if items[0] == 'Log' and not log_scale:
                new_label = ' '.join(items[1:])
            elif items[0] != 'Log' and log_scale:
                new_label = ' '.join(['Log'] + items)
        return new_label

    def setLabel(self, lbl):
        self.stacked_label = lbl

    def scalingUpdated(self, val):
        ''' Updates minBox and maxBoxes when color scaling changed '''
        if val:
            prefix = '10^'
            grid = self.spec.get_grid()
            minVal = np.log10(np.min(grid[grid>0]))
            maxVal = np.log10(np.max(grid[grid>0]))
            bounds = (-100, 100)
        else:
            prefix = ''
            minVal, maxVal = self.spec.get_grid_range()
            bounds = (-1e16, 1e16)

        for box in [self.minBox, self.maxBox]:
            box.setPrefix(prefix)
            box.setMinimum(bounds[0])
            box.setMaximum(bounds[1])

        self.minBox.setValue(minVal)
        self.maxBox.setValue(maxVal)

    def setupRangeLt(self):
        frame = QtWidgets.QGroupBox('Value Range: ')
        layout = QtWidgets.QGridLayout(frame)

        # Set up default checkbox
        self.defaultRngBox = QtWidgets.QCheckBox(' Default Values')
        layout.addWidget(self.defaultRngBox, 0, 0, 1, 2)

        # Set up min and max spinboxes
        lbls, boxes = [], []
        row = 1
        for label in ['Min: ', 'Max: ']:
            lbl = QtWidgets.QLabel(label)
            lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
            box = QtWidgets.QDoubleSpinBox()
            box.setMinimum(-360)
            box.setMaximum(360)
            lbls.append(lbl)
            boxes.append(box)

            layout.addWidget(lbl, row, 0, 1, 1)
            layout.addWidget(box, row, 1, 1, 1)
            row += 1

        self.minBox, self.maxBox = boxes

        # Set initial values for checkboxes
        minVal, maxVal = self.spec.get_value_range()
        if self.spec.log_color_scale():
            for box in boxes:
                box.setPrefix('10^')

        self.minBox.setValue(minVal)
        self.maxBox.setValue(maxVal)

        # Connect default checkbox to toggleElems for min/max spinboxes
        toggleFunc = functools.partial(self.toggleElems, lbls+boxes)
        self.defaultRngBox.toggled.connect(toggleFunc)

        # Toggle default box if no specific range is set
        if self.spec.val_range is None:
            self.defaultRngBox.setChecked(True)

        return frame

    def toggleElems(self, elems, val):
        for elem in elems:
            elem.setEnabled(not val)

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

class SpectraLineEditorUI():
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

    def evalExpr(self, exprStr, tO, tE):
        # Attempt to evaluate expression, print error if an exception occurs
        from .simpleCalculations import simpleCalc
        rng = (tO, tE)
        tool = simpleCalc(None, self.window)
        result = tool.evaluate(exprStr, rng)
        if result is None or result[0] is None:
            return None

        # Extract values from data
        name, data = result

        # Reshape constants to match times length
        if isinstance(data, (float, int)):
            times = np.array([tO, tE])
            data = np.array([data, data])
        else:
            times, data = data.value()

        return (times, data)

    def addToPlot(self):
        if not self.keepMode():
            self.clearPlot()
        tO, tE = self.spectraFrame.getTimeRange()
        exprStr = self.ui.textBox.toPlainText()
        if exprStr == '':
            return

        result = self.evalExpr(exprStr, tO, tE)
        if result is None: # Return + print error message for invalid expression
            self.ui.statusBar.showMessage('Error: Invalid operation')
            return

        times, data = result

        # Extract line settings from UI and times to use for data
        arbDstr = self.window.DATASTRINGS[0]
        color = self.ui.lineColor
        width = self.ui.lineWidth.value()
        lineStyle = self.ui.lineStyles[self.ui.lineStyle.currentText()]

        # Create line from user input and add it to the plot item
        lineItem = self.createLine(data, times, color, lineStyle, width)
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
        clrDialog = QtGui.QColorDialog(self)
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

        # Linked plot and menu toggle
        self.plot = None
        self.label = None
        self.editor = None
        self.menuEnabled = False

        # Set default contents spacing/margins
        super().__init__(parent)
        self.layout.setVerticalSpacing(0)
        self.layout.setHorizontalSpacing(0)
        self.layout.setContentsMargins(0, 2, 0, 0)

        self.addItem(self.colorBar, 0, 0, 1, 1)
        self.addItem(self.axisItem, 0, 1, 1, 1)

        self.setMinimumWidth(10)

    def getColorBar(self):
        return self.colorBar

    def getAxis(self):
        return self.axisItem

    def getPlot(self):
        return self.plot

    def setPlot(self, plot):
        self.plot = plot

    def getLabel(self):
        return self.label

    def setLabel(self, label):
        self.label = label

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
            left = 0
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

    def resetTickSpacing(self):
        self.axisItem.resetTickSpacing()

    def enableMenu(self, val=True):
        ''' Set menu and cursor '''
        self.menuEnabled = val
        if val:
            self.setCursor(QtCore.Qt.PointingHandCursor)
        else:
            self.setCursor(QtCore.Qt.Arrow)

    def mouseClickEvent(self, ev):
        leftClick = ev.button() == QtCore.Qt.LeftButton

        # Open editor if enabled
        if leftClick and self.menuEnabled:
            # Create new editor
            if not self.editor:
                self.editor = GradEditor(self)
                self.destroyed.connect(self.editor.close)
            else:
                # Bring previous editor to front
                self.editor.raise_()
            self.editor.show()

class DynamicAnalysisTool():
    fftBins = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 
        16384, 32768, 65536, 131072]
    def __init__(self):
        self.lineTool = None
        self.maskTool = None
        self.savedLineInfo = None
        self.lineHistory = set()
        self.lineInfoHist = set()
        self.fftBinBound = DynamicAnalysisTool.fftBins[-1]*4

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

    def splitDataSegments(self, times, data, interval, shift, onedim=False):
        ''' Splits data into segments based on the fft_interval and fft_shift

            Returns time_stops and data_segments as a tuple
        '''
        # Get length of data
        n = len(data) if onedim else len(data[0])

        # Iterate over indices with shift as the step size
        time_stops = []
        data_segs = []
        for i in range(0, n - interval + 1, shift):
            # Get the start/end indices and extract data segment
            # and save start time to time stops
            start = i
            stop = i + interval
            time_stops.append(times[start])

            if onedim:
                seg = data[start:stop]
            else:
                seg = data[:,start:stop]
            data_segs.append(seg.copy())

        # Add final time stop to the end
        stop = min(start+shift, len(times)-1)
        time_stops.append(times[stop])

        return (time_stops, data_segs)

    def setParams(self, fftParams, scale, detrend):
        # Set FFT parameters
        fftBoxes = [self.ui.fftInt, self.ui.fftShift, self.ui.bwBox]
        for box, val in zip(fftBoxes, list(fftParams)):
            box.setValue(val)

        # Set detrend mode and plot scale
        self.setDetrendMode(detrend)
        self.setAxisScaling(scale)

    def getTimeRange(self):
        minTime, maxTime = self.window.getSelectionStartEndTimes()
        return (minTime, maxTime)

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

        interval = self.guess_fft_param(nPoints)

        self.ui.fftInt.setValue(interval)
        overlap = int(interval/4)
        self.ui.fftShift.setValue(overlap)

    def guess_fft_param(self, npoints):
        bins = DynamicAnalysisTool.fftBins
        interval = max(min(npoints, 10), int(npoints*0.025))
        if npoints > bins[4]:
            index = bisect.bisect(bins, interval)
            if index < len(bins):
                interval = bins[index]
        return interval

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

            # Extract line info + generate the new line and add it to the plot
            expr, color, width, style = lineInfo
            lineEditor = SpectraLineEditor(self, self.window, (a,b))
            tO, tE = self.getTimeRange()
            (times, dta) = lineEditor.evalExpr(expr, tO, tE)
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
            specData = plt.getSpecData()[0]
            specCopy = copy(specData)

            # Remove analysis from plot name to make it shorter
            name = specCopy.get_name()
            if 'Analysis' in name:
                name = ' '.join([elem.strip(' ') for elem in name.split('Analysis')])
            specCopy.set_name(name)

            # Add spectrogram to main plot grid and close window
            self.window.add_spectrogram(specCopy, name)
            self.close()

class SpectraLegend(GradLegend):
    def __init__(self, offsets=(31, 48)):
        GradLegend.__init__(self)
        topOff, botOff = offsets
        self.setOffsets(topOff, botOff)
        self.logMode = False
        self.setMinimumWidth(10)
        self.setMaximumWidth(70)
        self.setMaximumHeight(1e5)
        self.setMinimumHeight(10)

    def getCopy(self):
        newLegend = SpectraLegend()
        newLegend.setRange(self.getGradient(), self.getValueRange())
        return newLegend

    def setLogMode(self, logMode):
        self.logMode = logMode
        if logMode:
            self.setTickSpacing(1, 0.5)
        else:
            self.setTickSpacing(None, None)

        # Update label if there is one linked to this gradlegend
        label = self.getLabel()
        if label:
            labels = label.getLabelText()
            if len(labels) == 0:
                return

            top_label = labels[0]
            new_top_label = self.update_log_label(top_label, logMode)
            labels[0] = new_top_label
            label.setupLabels(labels)

    def update_log_label(self, label, log_scale):
        items = label.split(' ')
        new_label = label
        if len(items) > 0:
            if items[0] == 'Log' and not log_scale:
                new_label = ' '.join(items[1:])
            elif items[0] != 'Log' and log_scale:
                new_label = ' '.join(['Log'] + items)
        return new_label

    def logModeSetting(self):
        return self.logMode

class SpectraGridItem(pg.GraphicsObject):
    '''
    Grid version of a set of SpectraLine items; Optimized for performance
    '''
    def __init__(self, freqs, colors, times, window=None, *args, **kargs):
        # Data bounds
        self.x_bound = times[[0, -1]]
        self.y_bound = freqs[[0, -1]]

        # Used to update window's status bar w/ the clicked value if passed
        self.window = window
        self.clickable = True
        self.spec = None
        self.logmode = False

        # Color and rect tuples to draw on plot
        shape = np.array(freqs).shape
        if len(shape) > 1 and shape[1] > 0:
            self.paths = list(self.setupMultiBinPath(freqs, colors, times))
            self.y_bound = [np.min(freqs), np.max(freqs)]
        else:
            self.paths = list(self.setupPath(freqs, colors, times))

        # Caching purposes
        self._boundingRect = None
        self._mapped_grid = None
    
        super().__init__()

        self.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)

    def cached_mapped_grid(self):
        ''' Returns a cached version of the mapped grid '''
        if self._mapped_grid is None and self.spec is not None:
            self._mapped_grid = self.spec.get_mapped_grid()
        return self._mapped_grid

    def setLogMode(self, x=False, val=True):
        if self.logmode != val:
            self.remap_paths(val)
            self.logmode = val

    def remap_paths(self, val):
        ''' Update rects when y scaling mode is updated '''
        for color, rect in self.paths:
            top, bottom = rect.top(), rect.bottom()
            if val: # Log scale
                top, bottom = np.log10(top), np.log10(bottom)
            else: # Linear scale
                top, bottom = top ** 10, bottom ** 10
            rect.setTop(top)
            rect.setBottom(bottom)
        self.prepareGeometryChange()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if ax == 0:
            b = self.x_bound
        else:
            b = self.y_bound
            if self.logmode:
                b = np.log10(self.y_bound)
        return b

    def pixelPadding(self):
        return 0

    def boundingRect(self):
        if self._boundingRect is None:
            (xmn, xmx) = self.dataBounds(ax=0)
            if xmn is None or xmx is None:
                return QtCore.QRectF()
            (ymn, ymx) = self.dataBounds(ax=1)
            if ymn is None or ymx is None:
                return QtCore.QRectF()

            px = py = 0.0
            pxPad = self.pixelPadding()
            if pxPad > 0:
                # determine length of pixel in local x, y directions
                px, py = self.pixelVectors()
                try:
                    px = 0 if px is None else px.length()
                except OverflowError:
                    px = 0
                try:
                    py = 0 if py is None else py.length()
                except OverflowError:
                    py = 0

                # return bounds expanded by pixel size
                px *= pxPad
                py *= pxPad
            self._boundingRect = QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)
        return self._boundingRect

    def setupPath(self, freqs, colors, times):
        # Creates subpath for each rect in the grid
        for r in range(0, len(freqs)-1):
            y0 = freqs[r]
            y1 = freqs[r+1]
            height = y1 - y0
            row_colors = colors[r]
            for c in range(0, len(times)-1):
                x0 = times[c]
                x1 = times[c+1]
                color = row_colors[c]

                # Upper left corner
                rect = QtCore.QRectF(x0, y0, x1-x0, height)
                yield (color, rect)

    def setupMultiBinPath(self, freqs, colors, times):
        # Iterate over each frequency column and data column
        pt = QtCore.QPointF(0, 0)
        freqRows = np.array(freqs).T

        for i in range(0, len(freqRows) - 1):
            freqs0 = freqRows[i]
            freqs1 = freqRows[i+1]
            heights = freqs1 - freqs0

            for j in range(0, len(times) - 1):
                x0 = times[j]
                x1 = times[j+1]

                y0 = freqs0[j]
                y1 = freqs1[j]

                color = colors[i][j]

                rect = QtCore.QRectF(x0, y0, x1 - x0, heights[j])
                yield (color, rect)

    def paint(self, p, opt, widget):
        if len(self.paths) == 0:
            return

        p.setRenderHint(p.Antialiasing, False)

        # Draws filled rects for every point using designated colors
        for color, rect in self.paths:
            p.fillRect(rect, color)

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

        # Additional matplotlib gradient information
        self.cmap = 'jet'

        # Parameters used in calculations
        self.param_vals = {}

    def values(self):
        ''' Returns the x, y, grid values used to plot the spectrogram '''
        y, x = self.get_bins(padded=True)
        grid = self.get_grid()
        return x, y, grid

    def params(self):
        return self.param_vals

    def single_y_bins(self):
        shape = self.y_bins.shape
        if len(shape) <= 1:
            return True
        else:
            return False

    def print_values(self):
        print ('Value Range:', self.val_range)
        print ('log_color:', self.log_color)
        print ('mask_info:', self.mask_info)
        print ('log_scale:', self.log_scale)
        print ('grad_range:', self.grad_range)
        print ('gradient_stops:', self.gradient_stops)
        print ('grid_range:', self.grid_range)

    def get_ranges(self):
        y, x = self.get_bins()
        x_range = (np.min(x), np.max(x))
        y_range = (np.min(y), np.max(y))
        return x_range, y_range

    def get_name(self):
        return self.name

    def get_labels(self):
        return self.y_label, self.legend_label

    def get_bins(self, padded=True):
        ''' Padded arg specifies whether to pad bins if # rows = # grid rows
            or # cols = # grid cols
        '''
        y_bins = self.y_bins
        x_bins = self.x_bins

        # Pad y bins if padded=True
        if padded:
            grid_shape = self.grid.shape
            y_shape = y_bins.shape
            padded_y = False

            # If # of y bins == number of rows in grid
            if len(y_shape) > 1 and y_shape[1] == grid_shape[0]:
                padded_y = True
                y_diff = y_bins[:,1] - y_bins[:,0]
                last_y = y_bins[:,0] - y_diff
                if sum(last_y < 0):
                    last_y = y_bins[:,0] / 2
                last_y = np.reshape(last_y, (len(last_y), 1))
            elif len(y_shape) == 1 and len(y_bins) == grid_shape[0]:
                padded_y = True
                y_diff = y_bins[1] - y_bins[0]
                last_y = y_bins[0] - y_diff
                if last_y < 0:
                    last_y = y_bins[0] / 2
                last_y = [last_y]

            if padded_y:
                y_bins = np.hstack([last_y, y_bins])

            if padded_y and len(y_shape) > 1 and y_shape[0] == grid_shape[1]:
                bottom_row = y_bins[-1]
                bottom_row = np.reshape(bottom_row, (1, len(bottom_row)))
                y_bins = np.vstack([y_bins, bottom_row])

            if len(x_bins) == len(self.grid[0]):
                x_end = x_bins[-1] + (x_bins[-2] - x_bins[-1])
                x_bins = np.hstack([x_bins, x_end])

        return y_bins, x_bins

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

    def cyclic_grad(self):
        return self.gradient_stops[0][1] == self.gradient_stops[-1][1]

    def get_mapped_grid(self, color_rng=None):
        grid = np.array(self.grid)
        if color_rng is None:
            color_rng = self.val_range

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

        # Mask out any time gaps with white
        y_bins, x_bins = self.get_bins()
        diff = np.mean(np.diff(x_bins))
        gap_indices = find_gaps(x_bins, diff*2)
        for index in gap_indices:
            mappedGrid[:,index-1] = (255, 255, 255, 255)

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
        self.log_color = val

    def set_val_range(self, rng):
        self.val_range = rng

    def set_y_label(self, lbl):
        self.y_label = lbl

    def set_y_log_scale(self, val):
        self.log_scale = val

    def set_legend_label(self, lbl):
        self.legend_label = lbl

    def set_name(self, name):
        self.name = name

    def set_data(self, y, grid, x):
        self.y_bins = y
        self.x_bins = x
        self.grid = grid
        self.grid_range = None

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
