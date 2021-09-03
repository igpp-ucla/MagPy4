from ..dispwidgets.layouttools import BaseLayout
from . import waveanalysis
from ..plotbase import MagPyPlotItem, StackedAxisLabel
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import numpy as np
from scipy import ndimage

class ToolSpecificLt(BaseLayout):
    def __init__(self):
        self.logMode = False
        super().__init__()
    
    def getMaskLt(self, valRng, logMode=False, frame=None):
        if frame:
            layout = QtWidgets.QGridLayout(frame)
        else:
            layout = QtWidgets.QGridLayout()

        # Set up boxes for specifying mask ranges
        self.minMaskCheck = QtWidgets.QCheckBox(' Greater than: ')
        self.maxMaskCheck = QtWidgets.QCheckBox(' Less than: ')

        self.minMaskBox = QtWidgets.QDoubleSpinBox()
        self.maxMaskBox = QtWidgets.QDoubleSpinBox()

        # Op box if both are selected
        self.subMaskOp = QtWidgets.QComboBox()
        self.subMaskOp.addItems(['AND', 'OR'])

        layout.addWidget(self.minMaskCheck, 0, 0, 1, 1)
        layout.addWidget(self.minMaskBox, 0, 1, 1, 1)
        layout.addWidget(self.maxMaskCheck, 1, 0, 1, 1)
        layout.addWidget(self.maxMaskBox, 1, 1, 1, 1)
        self.opLbl = self.addPair(layout, 'Mask Operation: ', self.subMaskOp, 2, 0, 1, 1)
        for elem in [self.opLbl, self.subMaskOp]:
            elem.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Set box settings
        minVal, maxVal = valRng
        if logMode:
            for box in [self.minMaskBox, self.maxMaskBox]:
                box.setPrefix('10^')
            self.logMode = True

        for box in [self.minMaskBox, self.maxMaskBox]:
            box.setMinimum(minVal)
            box.setMaximum(maxVal)
            box.setDecimals(3)

        # Hide op box when both are not set        
        for chk in [self.minMaskCheck, self.maxMaskCheck]:
            chk.toggled.connect(self.hideOpBox)
            chk.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.hideOpBox()

        return layout

    def getMaskRanges(self):
        # If min/max boxes are checked, return the box value or return None
        # otherwise
        if not self.minMaskCheck.isChecked():
            minVal = None
        else:
            minVal = self.minMaskBox.value()

        if not self.maxMaskCheck.isChecked():
            maxVal = None
        else:
            maxVal = self.maxMaskBox.value()

        # Scale if ranges are logarithmic
        if self.logMode:
            minVal = 10 ** minVal if minVal is not None else minVal
            maxVal = 10 ** maxVal if maxVal is not None else maxVal

        # Also return the binary operation to apply between the two
        # masks if both are set
        opVal = self.subMaskOp.currentText()

        return minVal, maxVal, opVal

    def hideOpBox(self):
        # Show binary op combobox only if both min and max boxes are checked
        minChecked = self.minMaskCheck.isChecked()
        maxChecked = self.maxMaskCheck.isChecked()
        if minChecked and maxChecked:
            self.subMaskOp.setVisible(True)
            self.opLbl.setVisible(True)
        else:
            self.subMaskOp.setVisible(False)
            self.opLbl.setVisible(False)

    def getVarInfo(self):
        return None

class MaskToolUI(BaseLayout):
    def setupUI(self, maskFrame, plotTool, plotType):
        self.maskFrame = maskFrame
        maskFrame.setWindowTitle('Mask Tool')
        layout = QtWidgets.QGridLayout(maskFrame)
        self.layout = layout

        # Get default min/max values to use in the mask value range setter
        minVal, maxVal = 0, 1
        logMode = False
        if plotType == 'Spectra':
            minVal = plotTool.ui.valueMin.minimum()
            maxVal = plotTool.ui.valueMax.maximum()
            logMode = True
        elif plotType in ['Coherence', 'Phase']:
            if plotType == 'Coherence':
                minVal, maxVal = 0, 1.0
            else:
                minVal, maxVal = -180, 180
        else:
            valRange = plotTool.defParams[plotType][0]
            if valRange is not None:
                minVal, maxVal = valRange
            else:
                minVal, maxVal = -18, 18
            if plotType in plotTool.plotGroups['Power']:
                logMode = True
        valRange = (minVal, maxVal)

        # Set up mask values calculator
        groupFrame = QtWidgets.QGroupBox('Mask Values')
        self.toolMaskLt = ToolSpecificLt()
        toolMaskSubLt = self.toolMaskLt.getMaskLt(valRange, logMode, groupFrame)

        # Mask properties layout
        maskPropLt = self.setupMaskSettingsLt()

        # Update button
        self.updateBtn = QtWidgets.QPushButton(' Plot ')

        settingsLt = QtWidgets.QHBoxLayout()
        for lt in [groupFrame, maskPropLt, self.updateBtn]:
            settingsLt.addWidget(lt)
        
        settingsLt.setAlignment(self.updateBtn, QtCore.Qt.AlignBottom)

        layout.addLayout(settingsLt, 0, 0, 1, 1, QtCore.Qt.AlignTop|QtCore.Qt.AlignLeft)
        layout.setRowStretch(0, 0)

        # Plot graphics grid
        self.glw = self.getGraphicsGrid()
        self.gview.setVisible(False)
        layout.addWidget(self.gview, 1, 0, 1, 1)

    def setupMaskSettingsLt(self):
        frame = QtWidgets.QGroupBox('Mask Properties')
        frame.resize(100, 50)
        layout = QtWidgets.QGridLayout()
        layout.setHorizontalSpacing(10)

        wrapLt = QtWidgets.QVBoxLayout(frame)
        wrapLt.addLayout(layout)
        wrapLt.setAlignment(QtCore.Qt.AlignTop)

        # Set up button for setting mask color
        self.colorBox = QtWidgets.QPushButton()
        self.colorBox.clicked.connect(self.openColorSelect)

        # Default mask color is white
        self.maskColor = (255, 255, 255)
        self.setMaskColor(QtGui.QColor(255, 255, 255))

        colorLt = QtWidgets.QGridLayout()
        colorLt.setContentsMargins(0, 0, 0, 0)
        self.addPair(colorLt, 'Color:  ', self.colorBox, 0, 0, 1, 1)

        # Mask outline box
        self.outlineCheck = QtWidgets.QCheckBox(' Outline Only')

        # Filtering check box
        filterLt = QtWidgets.QVBoxLayout()
        self.filterCheck = QtWidgets.QCheckBox(' Apply Gaussian Filter')

        # Sigma value box, frame, label, and settings
        self.sigmaBox = QtWidgets.QDoubleSpinBox()
        self.sigmaBox.setMaximum(3)
        self.sigmaBox.setDecimals(3)
        self.sigmaBox.setValue(1)
        self.sigmaBox.setMinimum(0.001)

        sigmaFrm = QtWidgets.QFrame()
        sigmaLt = QtWidgets.QGridLayout(sigmaFrm)
        sigmaLt.setContentsMargins(24, 0, 0, 0)
        self.addPair(sigmaLt, 'Sigma: ', self.sigmaBox, 0, 0, 1, 1)

        # Add sub layouts and widgets into frame
        layout.addLayout(colorLt, 0, 0, 1, 1)
        layout.addWidget(self.outlineCheck, 1, 0, 1, 1)
        layout.addWidget(self.filterCheck, 0, 1, 1, 1)
        layout.addWidget(sigmaFrm, 1, 1, 1, 1)

        return frame

    def openColorSelect(self):
        # Open color selection dialog and connect to line color update function
        clrDialog = QtGui.QColorDialog(self.maskFrame)
        clrDialog.show()
        clrDialog.colorSelected.connect(self.setMaskColor)

    def setMaskColor(self, color):
        self.maskColor = color.getRgb()[0:3]
        styleSheet = "* { background:" + color.name() + " }"
        self.colorBox.setStyleSheet(styleSheet)
        self.colorBox.show()

    def adjustWindowSize(self):
        # Resize window after plot is made visible
        if not self.gview.isVisible():
            self.gview.setVisible(True)
            self.layout.invalidate()
            size = self.maskFrame.size()
            height, width = size.height(), size.width()
            height = min(height+750, 750)
            width = min(950, width+950)
            self.maskFrame.resize(width, height)

class MaskTool(QtWidgets.QFrame):
    def __init__(self, toolFrame, plotType, parent=None):
        super().__init__(parent)
        self.tool = toolFrame
        self.window = toolFrame.window
        self.plotType = plotType
        self.ui = MaskToolUI()
        self.ui.setupUI(self, toolFrame, plotType)

        # Make a list of plot types that don't have a log color scale
        waveObj = waveanalysis.DynamicWave(self.window)
        self.linearColorPlots = ['Coherence', 'Phase']
        self.linearColorPlots += waveObj.plotGroups['Ellipticity']
        self.linearColorPlots += waveObj.plotGroups['Angle']
    
        # Set up default plot info for dynamic spectra, coherence, and phase plots
        self.defaultPlotInfo = {
            'Spectra': (None, 'Log Power', 'nT^2/Hz'),
            'Coherence': ((0, 1), 'Coherence', None),
            'Phase': ((-180, 180), 'Angle', 'Degrees')
        }

        # Copy parameter info from Wave Analysis object
        waveObj = waveanalysis.DynamicWave(self.window)
        waveObj.numThreads = 1
        for kw in waveObj.defParams.keys():
            self.defaultPlotInfo[kw] = waveObj.defParams[kw]

        self.ui.updateBtn.clicked.connect(self.update)

    def getVarInfos(self):
        varInfo = self.ui.toolMaskLt.getVarInfo()
        return varInfo

    def getMaskRanges(self):
        maskRng = self.ui.toolMaskLt.getMaskRanges()
        return maskRng

    def getColorRng(self, grid):
        # Get the range of values to map colors to for each tool type
        if self.plotType == 'Spectra':
            colorRng = self.tool.getGradRange()
            if colorRng is None:
                colorRng = (np.min(grid[grid>0]), np.max(grid[grid>0]))
        elif self.plotType in ['Coherence', 'Phase']:
            if self.plotType == 'Coherence':
                colorRng = (0, 1.0)
            else:
                colorRng = (-180, 180)
        else:
            colorRng = self.tool.getColorRng(self.plotType, grid)
        
        return colorRng

    def update(self):
        self.ui.adjustWindowSize()

        # Get plot info and parameters from main tool
        grid, freqs, times = self.getValueGrid()
        logScale = self.tool.getAxisScaling() == 'Logarithmic'
        varInfo = self.tool.getVarInfo()
        colorRng = self.getColorRng(grid)

        # Generate mask
        maskRng = self.ui.toolMaskLt.getMaskRanges()
        maskInfo = self.createMask(grid, maskRng)

        # Generate the plot and arrange the items in the plot layout
        plt = self.getPlotItem(grid, freqs, times, logScale, colorRng, maskInfo)
        lbls = self.getLabels(varInfo, logScale)
        self.setupGrid(plt, lbls, times)

        # Save state and add any plotted lines
        self.plt = plt
        for line in self.tool.lineHistory:
            self.addLineToPlot(line)
    
    def addLineToPlot(self, line):
        # Make a copy of the line item and add it to the plot
        pen = line.opts['pen']
        self.plt.plot(line.xData, line.yData, pen=pen)

    def setupGrid(self, plt, lbls, times):
        # Get the gradient legend from the plot, using log tick marks where appropr.
        if 'Power' in self.plotType or 'Spectra' == self.plotType:
            gradLegend = plt.getGradLegend(logMode=True)
        else:
            gradLegend = plt.getGradLegend()
        gradLegend.setBarWidth(38)

        # Set custom gradient legend tick spacing
        spacing = self.tool.getGradTickSpacing(self.plotType)
        if spacing is not None:
            major, minor = spacing
            gradLegend.setTickSpacing(major, minor)

        # Set up plot labels
        title, axisLbl, legendLbl = lbls
        plt.setTitle(title, size='13pt')
        plt.getAxis('left').setLabel(axisLbl)
    
        # Time info
        timeInfo = self.tool.getTimeInfoLbl((times[0], times[-1]))

        self.ui.glw.clear()
        self.ui.glw.addItem(plt, 0, 0, 1, 1)
        self.ui.glw.addItem(gradLegend, 0, 1, 1, 1)
        self.ui.glw.addItem(legendLbl, 0, 2, 1, 1)
        self.ui.glw.addItem(timeInfo, 1, 0, 1, 3)

    def getLabels(self, varInfo, logScale):
        if self.plotType == 'Spectra':
            lbls = self.tool.getLabels(varInfo, logScale)
        elif self.plotType in ['Coherence', 'Phase']:
            lbls = self.tool.getLabels(self.plotType, varInfo, logScale)
        else:
            lbls = self.tool.getLabels(self.plotType, logScale)
        return lbls

    def getValueGrid(self):
        if self.plotType in ['Coherence', 'Phase']:
            freqs, times, cohGrid, phaGrid = self.tool.lastCalc
            if self.plotType == 'Coherence':
                grid = cohGrid
            else:
                grid = phaGrid
        else:
            times, freqs, grid = self.tool.lastCalc
        
        return grid, freqs, times

    def createMask(self, grid, maskRng):
        # If filter box is checked, apply a Gaussian filter to the grid before
        # the masks are generated
        filtered = self.ui.filterCheck.isChecked()
        if filtered:
            sigma = self.ui.sigmaBox.value()
            grid = ndimage.gaussian_filter(grid, sigma=sigma)

        # Extract parameters
        alphaMin, alphaMax, alphaOp = maskRng

        # Build masks that select everything by default
        alphaMask = np.full(grid.shape, True)

        # Apply mask if min/max value is set for each grid
        if alphaMin is not None:
            alphaMask = np.logical_and(alphaMask, (grid > alphaMin))

        if alphaMax is not None:
            # If both masks are set, apply the binary operation to the masks here
            if alphaMin and alphaOp == 'OR':
                alphaMask = np.logical_or(alphaMask, (grid < alphaMax))
            else:
                alphaMask = np.logical_and(alphaMask, (grid < alphaMax))

        maskOutline = self.ui.outlineCheck.isChecked()

        # If no masks were applied, unmask everything
        if alphaMin is None and alphaMax is None:
            alphaMask = np.logical_not(alphaMask)
            maskOutline = False

        return alphaMask, self.getMaskColor(), maskOutline

    def getMaskColor(self):
        return self.ui.maskColor

    def extendFreqs(self, freqs):
        # Get lower bound for frequencies on plot
        diff = abs(freqs[1] - freqs[0])
        lowerFreqBnd = freqs[0] - diff
        if lowerFreqBnd == 0 and self.ui.scaleModeBox.currentText() == 'Logarithmic':
            lowerFreqBnd = freqs[0] - diff/2
        freqs = np.concatenate([[lowerFreqBnd], freqs])
        return freqs

    def getPlotItem(self, grid, freqs, times, logScale, colorRng, maskInfo):
        # Determine if color map should interpret the values on a log scale
        if self.plotType in self.linearColorPlots:
            logColorScale = False
        else:
            logColorScale = True

        # Wrapped colors should be used for phase plots
        # A regular spectrogram plot item should be used otherwise
        if self.plotType == 'Phase':
            plt = MagPyPlotItem(self.window.epoch)
        else:
            plt = MagPyPlotItem(self.window.epoch)

        # Get the lower bound for the frequencies and generate the plot
        # from the grid values and mask info
        freqs = self.extendFreqs(freqs)
        plt.createPlot(freqs, grid, times, colorRng, logColorScale, maskInfo=maskInfo)

        return plt