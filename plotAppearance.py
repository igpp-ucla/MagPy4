from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from datetime import datetime, timedelta
import numpy as np

import pyqtgraph as pg
import functools

class PlotAppearanceUI(object):
    def setupUI(self, Frame, window, plotsInfo):
        Frame.setWindowTitle('Plot Appearance')
        Frame.resize(300, 200)
        layout = QtWidgets.QGridLayout(Frame)
        self.layout = layout

        # Font size label setup
        self.titleSzLbl = QtWidgets.QLabel('Title size: ')
        self.axisLblSzLbl = QtWidgets.QLabel('Axis label size: ')
        self.tickLblSzLbl = QtWidgets.QLabel('  Tick label size: ')

        # Title, axis label, and tick label spinboxes setup
        self.titleSzBox = QtWidgets.QSpinBox()
        self.titleSzBox.setMinimum(5)
        self.titleSzBox.setMaximum(30)

        self.axisLblSzBox = QtWidgets.QSpinBox()
        self.axisLblSzBox.setMinimum(5)
        self.axisLblSzBox.setMaximum(25)

        self.tickLblSzBox = QtWidgets.QSpinBox()
        self.tickLblSzBox.setMinimum(5)
        self.tickLblSzBox.setMaximum(25)

        layout.addWidget(self.titleSzLbl, 0, 0, 1, 1)
        layout.addWidget(self.axisLblSzLbl, 1, 0, 1, 1)
        layout.addWidget(self.tickLblSzLbl, 1, 2, 1, 1)

        layout.addWidget(self.titleSzBox, 0, 1, 1, 1)
        layout.addWidget(self.axisLblSzBox, 1, 1, 1, 1)
        layout.addWidget(self.tickLblSzBox, 1, 3, 1, 1)

        for lbl in [self.titleSzLbl, self.axisLblSzLbl, self.tickLblSzLbl]:
            lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Set up UI for setting plot trace colors, line style, thickness, etc.
        tracePropFrame = QtWidgets.QGroupBox('Line Properties')
        tracePropFrame.setAlignment(QtCore.Qt.AlignCenter)
        tracePropLayout = QtWidgets.QGridLayout(tracePropFrame)

        pltNum = 0 # Lists for storing interactive UI elements
        self.lineWidthBoxes = []
        self.lineStyleBoxes = []
        self.colorBoxes = []

        colNum, rowNum = 0, 0 # Layout ordering for excessive number of plots
        maxTracesPerCol = 3 * 3
        totTraces = 0

        for trcList in plotsInfo:
            # Group plot traces by plot number
            plotFrame = QtWidgets.QGroupBox('Plot '+str(pltNum + 1)+':')
            plotLayout = QtWidgets.QVBoxLayout(plotFrame)

            traceNum = 0
            for trcPen in trcList:
                traceLayout = QtWidgets.QHBoxLayout()
                label = QtWidgets.QLabel('Line '+str(traceNum + 1)+': ')

                # Create all elements for choosing line style
                styleLabel = QtWidgets.QLabel('  Style: ')
                lineStyle = QtWidgets.QComboBox()
                for t in ['Solid', 'Dashed', 'Dotted', 'DashDot']:
                    lineStyle.addItem(t)
                self.lineStyleBoxes.append((lineStyle, (pltNum, traceNum)))

                # Create all elements for choosing line thickness
                widthLabel = QtWidgets.QLabel('  Width: ')
                lineWidth = QtWidgets.QSpinBox()
                lineWidth.setMinimum(1)
                lineWidth.setMaximum(5)
                self.lineWidthBoxes.append((lineWidth, (pltNum, traceNum)))

                # Create elements for choosing line color
                colorLbl = QtWidgets.QLabel('Color: ')
                colorBtn = QtWidgets.QPushButton()
                colorBtn.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
                self.colorBoxes.append((colorBtn, (pltNum, traceNum)))

                # Add all elements to sublayout
                for e in [label, colorLbl, colorBtn, styleLabel, lineStyle,
                        widthLabel, lineWidth]:
                    traceLayout.addWidget(e)
                    traceLayout.setAlignment(e, QtCore.Qt.AlignBaseline)

                plotLayout.addLayout(traceLayout)
                traceNum += 1
            plotLayout.addStretch()

            tracePropLayout.addWidget(plotFrame, rowNum, colNum, 1, 1)
            pltNum += 1

            # Move to next column if max num of traces in row
            rowNum += 1
            totTraces += traceNum

            if totTraces >= maxTracesPerCol*(colNum+1):
                rowNum = 0
                colNum += 1

        layout.addWidget(tracePropFrame, 3, 0, 1, 4)

        # Add in button to open tick spacing window
        self.tickIntBtn = QtWidgets.QPushButton(' Adjust Tick Spacing... ')
        if colNum >= 1 and rowNum >= 1:
            layout.addWidget(self.tickIntBtn, 4, 0, 1, 2)
        else:
            layout.addWidget(self.tickIntBtn, 4, 1, 1, 2)

class PlotAppearance(QtGui.QFrame, PlotAppearanceUI):
    def __init__(self, window, plotItems, parent=None):
        super(PlotAppearance, self).__init__(parent)
        self.ui = PlotAppearanceUI()
        self.plotItems = plotItems
        self.window = window
        self.tickIntervals = None

        # Get plots' trace/label infos and use to setup/initialize UI elements
        plotsInfo = self.getPlotsInfo()
        self.ui.setupUI(self, window, plotsInfo)
        self.initVars(plotsInfo)

        # Connect buttons to functions
        self.ui.titleSzBox.valueChanged.connect(self.changeTitleSize)
        self.ui.axisLblSzBox.valueChanged.connect(self.changeAxisLblSize)
        self.ui.tickLblSzBox.valueChanged.connect(self.changeTickLblSize)
        self.ui.tickIntBtn.clicked.connect(self.openTickIntervals)

        # Connect line width modifiers to function
        for lw, indices in self.ui.lineWidthBoxes:
            pltNum, trcNum = indices
            pltPen, traceList = plotsInfo[pltNum][trcNum]
            lw.valueChanged.connect(functools.partial(self.updateLineWidth, lw, traceList))

        # Connect line style modifiers to function
        for ls, indices in self.ui.lineStyleBoxes:
            pltNum, trcNum = indices
            pltPen, traceList = plotsInfo[pltNum][trcNum]
            ls.currentIndexChanged.connect(functools.partial(self.updateLineStyle, ls, traceList))

        # Connect line color modifiers to function
        for cs, indices in self.ui.colorBoxes:
            pltNum, trcNum = indices
            pltPen, traceList = plotsInfo[pltNum][trcNum]
            cs.clicked.connect(functools.partial(self.openColorSelect, cs, traceList))

    def getPlotsInfo(self):
        # Creates list of per-plot lists containing tuples of pens and a list of
        # data items within the given plot that correspond to it
        # ex: [ [(pen1, [plotDataItem1...]), (pen2, [..])] , ... , ]
        plotsInfo = []
        for plt in self.plotItems:
            pltInfo = []
            uniqPltPens = [] # Create list of unique pens in current plot
            for pt in plt.listDataItems():
                pen = pt.opts['pen']
                if pen not in uniqPltPens:
                    # Add unseen pens to list and init its list of traces
                    uniqPltPens.append(pen)
                    pltInfo.append((pen, [pt]))
                else:
                    # Add current trace item to corresp. pen's trace list
                    index = uniqPltPens.index(pen)
                    penItemList = pltInfo[index][1]
                    penItemList.append(pt)
            plotsInfo.append(pltInfo)
        return plotsInfo

    def initVars(self, plotsInfo):
        # Initializes all values in UI elements with current plot properties
        if self.plotItems == None:
            return
        # Get attributes from last plot in set
        plt = self.plotItems[-1]

        # Get title font size to initialize spin box, disable if no title
        titleSize = plt.titleLabel.opts['size'][:-2] # Strip pts part of string
        if plt.titleLabel.text == '':
            self.ui.titleSzLbl.setEnabled(False)
            self.ui.titleSzBox.setEnabled(False)
        else:
            self.ui.titleSzBox.setValue(int(titleSize))

        # Initialize axis label font size, disable if no axis label
        self.ui.axisLblSzBox.blockSignals(True) # Prevent plot updates temporarily
        if plt.getAxis('bottom').label.toPlainText() == '':
            self.ui.axisLblSzBox.setEnabled(False)
            self.ui.axisLblSzLbl.setEnabled(False)
        elif 'font-size' in plt.getAxis('bottom').labelStyle:
            axisLblSize = plt.getAxis('bottom').labelStyle['font-size'][:-2]
            self.ui.axisLblSzBox.setValue(int(axisLblSize))
        else:
            self.ui.axisLblSzBox.setValue(11) # Default axis label font size
        self.ui.axisLblSzBox.blockSignals(False)

        # Initialize tick label font size
        self.ui.tickLblSzBox.blockSignals(True)
        axis = plt.getAxis('bottom')
        if axis.tickFont == None: # No custom font, use default
            self.ui.tickLblSzBox.setValue(11)
        else:
            tickFont = axis.tickFont
            tickSize = tickFont.pointSize()
            if tickSize < 0: # No point size set, use default
                tickSize = 11
            self.ui.tickLblSzBox.setValue(int(tickSize))
        self.ui.tickLblSzBox.blockSignals(False)

        traceNum = 0
        for pltGrp in plotsInfo:
            for pen, lst in pltGrp:
                # Initialize plot trace line styles
                style = pen.style()
                if style == QtCore.Qt.DashLine:
                    styleStr = 'Dashed'
                elif style == QtCore.Qt.DotLine:
                    styleStr = 'Dotted'
                elif style == QtCore.Qt.DashDotLine:
                    styleStr = 'DashDot'
                else:
                    styleStr = 'Solid'
                self.ui.lineStyleBoxes[traceNum][0].setCurrentText(styleStr)

                # Initialize plot trace line widths
                width = pen.width()
                self.ui.lineWidthBoxes[traceNum][0].setValue(width)

                # Initialize plot trace line colors
                color = pen.color()
                self.setButtonColor(self.ui.colorBoxes[traceNum][0], color)

                traceNum += 1

    def updateLineWidth(self, lw, ll, val):
        for pt in ll:
            pen = pt.opts['pen']
            pen.setWidth(val)
            pt.setPen(pen)
        self.setChangesPersistent(self.getPenList())

    def updateLineStyle(self, ls, ll):
        # Get style object from name
        styleStr = ls.currentText()
        if styleStr == 'Dashed':
            style = QtCore.Qt.DashLine
        elif styleStr == 'Dotted':
            style = QtCore.Qt.DotLine
        elif styleStr == 'DashDot':
            style = QtCore.Qt.DashDotLine
        else:
            style = QtCore.Qt.SolidLine

        # Update pens for selected plots
        for pt in ll:
            pen = pt.opts['pen']
            pen.setStyle(style)
            pt.setPen(pen)
        self.setChangesPersistent(self.getPenList())

    def openColorSelect(self, cs, ll):
        # Open color selection dialog and connect to line color update function
        clrDialog = QtWidgets.QColorDialog(self)
        clrDialog.show()
        clrDialog.colorSelected.connect(functools.partial(self.setLineColor, cs, ll))

    def setLineColor(self, cs, ll, color):
        # Update pen color of every trace item in ll corresp. to the original pen
        for pt in ll:
            pen = pt.opts['pen']
            pen.setColor(color)
            pt.setPen(pen)

        # Match the color selection button to the selected color
        self.setButtonColor(cs, color)
        # Set the title colors to match, if implemented
        self.adjustTitleColors(self.getPenList())
        self.setChangesPersistent(self.getPenList())

    def setButtonColor(self, cs, color):
        styleSheet = "* { background:" + color.name() + " }"
        cs.setStyleSheet(styleSheet)
        cs.show()

    # Placeholder func to be implemented if title colors must match line colors
    def adjustTitleColors(self, penList):
        pass

    # Placeholder func to be implemented to make line changes persistent
    # between changes to the plot traces
    def setChangesPersistent(self, penList):
        pass

    def getPenList(self):
        # Returns a list of the pen lists for each plot in plotItems
        pltInfos = self.getPlotsInfo()
        penList = []
        # For each plot, extract its pen list from its plotInfo
        for pltInfo in pltInfos:
            pltPens = []
            for pen, ll in pltInfo:
                pltPens.append(pen)
            penList.append(pltPens)
        return penList

    def changeTitleSize(self, val):
        for plt in self.plotItems:
            plt.titleLabel.setText(plt.titleLabel.text, size=str(val)+'pt')

    def changeAxisLblSize(self, val):
        # Update every plot's label sizes for every axis
        for plt in self.plotItems:
            for ax in [plt.getAxis('bottom'), plt.getAxis('left')]:
                if ax.label.toPlainText() == '': # Do not change size if no label
                    continue
                # Convert new value to apprp string and update label info
                sizeStr = str(val) + 'pt'
                ax.labelStyle = {'font-size':sizeStr}
                ax.label.setHtml(ax.labelString()) # Uses updated HTML string

    def changeTickLblSize(self, val):
        # Update every axes' tick label sizes
        for plt in self.plotItems:
            for axis in [plt.getAxis('left'), plt.getAxis('bottom')]:
                # Update font-size, using default if not previously set
                tickFont = axis.style['tickFont']
                if tickFont == None:
                    tickFont = QtGui.QFont()
                tickFont.setPointSize(val)
                axis.setStyle(tickFont=tickFont)
                axis.tickFont = tickFont

                # Adjust vert/horz spacing reserved for bottom ticks if necessary
                self.adjustTickHeights(axis, tickFont)

    def adjustTickHeights(self, axis, tickFont):
        # Adjust vertical spacing reserved for bottom ticks if necessary
        mets = QtGui.QFontMetrics(tickFont)
        ht = mets.boundingRect('AJOW').height() # Tall letters
        if ht > 18 and axis.orientation == 'bottom':
            axis.setStyle(tickTextOffset=5)
        elif axis.orientation == 'bottom':
            axis.setStyle(tickTextOffset=2)

    def openTickIntervals(self):
        self.closeTickIntervals()
        self.tickIntervals = TickIntervals(self.window, self.plotItems)
        self.tickIntervals.show()

    def closeTickIntervals(self):
        if self.tickIntervals:
            self.tickIntervals.close()
            self.tickIntervals = None

    def closeEvent(self, event):
        self.closeTickIntervals()
        self.close()

class MagPyPlotApp(PlotAppearance):
    def __init__(self, window, plotItems, parent=None):
        PlotAppearance.__init__(self, window, plotItems, parent)
        self.ui.layout.removeWidget(self.ui.titleSzBox)
        self.ui.layout.removeWidget(self.ui.titleSzLbl)
        self.ui.titleSzBox.deleteLater()
        self.ui.titleSzLbl.deleteLater()

    def adjustTitleColors(self, penList):
        self.window.pltGrd.adjustTitleColors(penList)
        self.window.plotTracePens = penList

    def setChangesPersistent(self, penList):
        # Update main window's current pen list
        self.window.plotTracePens = penList
        # Create new pen list to look through every time plots are rebuilt w/ plotData
        allPltStrs = self.window.lastPlotStrings
        customPens = []
        for pltStrs, pltPens in zip(allPltStrs, penList):
            pltCstmPens = []
            for (pltStr, en), pen in zip(pltStrs, pltPens):
                pltCstmPens.append((pltStr, en, pen))
            customPens.append(pltCstmPens)
        # Stores per-plot lists of (dstr, en, newPen) tuples for every trace
        self.window.customPens = customPens

    def openTickIntervals(self):
        self.closeTickIntervals()
        self.tickIntervals = MagPyTickIntervals(self.window, self.plotItems)
        self.tickIntervals.show()

class SpectraPlotApp(PlotAppearance):
    def __init__(self, window, plotItems, parent=None):
        PlotAppearance.__init__(self, window, plotItems, parent)

    def adjustTitleColors(self, penList):
        self.window.updateTitleColors(penList)

    def setChangesPersistent(self, penList):
        self.window.tracePenList = penList

    def adjustTickHeights(self, axis, tickFont):
        # Adjust horizontal spacing to account for numbers w/ superscripts
        mets = QtGui.QFontMetrics(tickFont)
        wdth = mets.averageCharWidth()
        # Default is 2, other values were found through testing
        if axis.orientation == 'left' and wdth > 11:
            axis.setStyle(tickTextOffset=13)
        elif axis.orientation == 'left' and wdth > 9:
            axis.setStyle(tickTextOffset=7)
        elif axis.orientation == 'left':
            axis.setStyle(tickTextOffset=2)

### Tick Spacing classes ###
class TickIntervalsUI(object):
    def setupUI(self, Frame, window, plotItems, links):
        Frame.setWindowTitle('Set Tick Spacing')
        Frame.resize(200, 100)
        layout = QtWidgets.QGridLayout(Frame)

        btmPlt = plotItems[-1]

        # Determine Y val with largest magnitude to set spinbox max for left axes
        yMax = 100
        for plt in plotItems:
            for pdi in plt.listDataItems():
                yMax = max(yMax, max(np.absolute(pdi.yData)))

        # Collect UI boxes/buttons and information about the corresp. axis
        self.intervalBoxes = []
        for name in ['left', 'bottom']:
            # Builds frames / UI elements for corresponding axis side
            self.buildAxisBoxes(layout, name, btmPlt, yMax, links)

    def getBoxName(self, axOrient, linkGrp):
        # Create name for linked axes if more than one link grp
        nameBase = 'Linked ' + axOrient[0].upper() + axOrient[1:] + ' Axes '
        boxName = nameBase + str([pltNum + 1 for pltNum in linkGrp])
        return boxName

    def setupBoxDefaults(self, ax, axType, box, name, yMax):
        # Set maximum interval for left axes
        if name == 'left' and axType == 'Regular':
            box.setMaximum(yMax)

        if axType == 'DateTime':
            # Determine initial time interval if previously set
            if ax.tickDiff is not None:
                hr = ax.tickDiff.seconds / 3600
                minutes = (ax.tickDiff.seconds % 3600)/60
                seconds = (ax.tickDiff.seconds % 60)
                prevTime = QtCore.QTime(hr, minutes, seconds)
                box.setTime(prevTime)
            else:
                # Otherwise, default to 10 minutes
                hrtime = QtCore.QTime(0,10,0)
                box.setTime(hrtime)
        else:
            # Set initial value for spinbox if previously set
            if ax.tickDiff is not None:
                box.setValue(ax.tickDiff)
            else:
                box.setValue(1)

    def buildAxisBoxes(self, layout, name, btmPlt, yMax, links):
        # For every linked axis group
        numLinks = len(links) if (name == 'left' and links is not None) else 1
        for i in range(0, numLinks):
            # Get axis, axis type, and create UI elements as apprp.
            ax = btmPlt.getAxis(name)
            axType = ax.axisType()

            # Create box label, add numbers corresp. to links if more than one grp
            boxName = name[0].upper() + name[1:] + ' Axis'
            if numLinks > 1 and name == 'left':
                if len(links[i]) > 1:
                    boxName = self.getBoxName(name, links[i])
                else:
                    boxName = boxName + ' ' + str(links[i][0]+1)

            # Builds the actual UI elements from info above
            axisFrame, box, applyBtn, defBtn = self.buildIntervalUI(boxName, axType)
            self.setupBoxDefaults(ax, axType, box, name, yMax)

            # Store this axe's info/UI elements and add frame to layout
            self.intervalBoxes.append((box, name, axType, applyBtn, defBtn))
            layout.addWidget(axisFrame)

    def buildIntervalUI(self, name, axType):
        # Initialize common UI elements
        axisFrame = QtWidgets.QGroupBox(name + ':')
        applyBtn = QtWidgets.QPushButton('Apply')
        defBtn = QtWidgets.QPushButton('Default')

        if axType == 'DateTime':
            # Create layout w/ timeEdit and label to describe time sections
            dateTimeLt = QtWidgets.QGridLayout(axisFrame)
            lbl = QtWidgets.QLabel('HH:MM:SS')
            intervalBox = QtWidgets.QTimeEdit()
            intervalBox.setDisplayFormat('HH:mm:ss')
            intervalBox.setMinimumTime(QtCore.QTime(0, 0, 1))
            dateTimeLt.addWidget(lbl, 0, 0, 1, 2)
            dateTimeLt.addWidget(intervalBox, 1, 0, 1, 1)
            dateTimeLt.addWidget(defBtn, 1, 1, 1, 1)
            dateTimeLt.addWidget(applyBtn, 1, 2, 1, 1)
        else:
            # Create spinbox and add elements to horizontal layout
            intervalLt = QtWidgets.QHBoxLayout(axisFrame)
            intervalBox = QtWidgets.QDoubleSpinBox()
            # Log spinbox should only take integers and is prefixed by '10^'
            if axType == 'Log':
                intervalBox = QtWidgets.QSpinBox()
                intervalBox.setPrefix('10^')
                intervalBox.setMinimum(1)
            else:
                intervalBox.setMinimum(0.01)
            intervalLt.addWidget(intervalBox)
            intervalLt.addWidget(defBtn)
            intervalLt.addWidget(applyBtn)
        return axisFrame, intervalBox, applyBtn, defBtn

class TickIntervals(QtGui.QFrame, TickIntervalsUI):
    def __init__(self, window, plotItems, parent=None, links=None):
        super(TickIntervals, self).__init__(parent)
        self.ui = TickIntervalsUI()
        self.ui.setupUI(self, window, plotItems, links)
        self.window = window
        self.plotItems = plotItems
        self.links = links

        # Connect 'apply' and 'default' buttons to appr functions
        for i, (intBox, name, axType, applyBtn, defBtn) in enumerate(self.ui.intervalBoxes):
            applyBtn.clicked.connect(functools.partial(self.applyChange, intBox, name, axType, i))
            defBtn.clicked.connect(functools.partial(self.setDefault, name, i))

    def setDefault(self, name, index):
        # Use all plots if no link settings, otherwise only update link grp
        if self.links is not None and name == 'left':
            pltIndices = self.links[index]
        else:
            pltIndices = [pltNum for pltNum in range(len(self.plotItems))]

        # Have each plot's corresp. axes reset its tick spacing
        for pltNum in pltIndices:
            plt = self.plotItems[pltNum]
            axis = plt.getAxis(name)
            axis.resetTickSpacing()

        self.additionalUpdates()

    def applyChange(self, intBox, name, axType, index):
        # Get spinbox value
        if axType == 'DateTime':
            # Convert QTime object to a timedelta object for DateTime axes
            value = intBox.time()
            formatStr = '%H:%M:%S'
            minBoxTime = datetime.strptime(QtWidgets.QTimeEdit().minimumTime().toString(), formatStr)
            valTime = datetime.strptime(value.toString(), formatStr)
            value = valTime - minBoxTime
        else:
            value = intBox.value()

        # Use all plots if no link settings, otherwise only update link grp
        if self.links is not None and name == 'left':
            pltIndices = self.links[index]
        else:
            pltIndices = [pltNum for pltNum in range(len(self.plotItems))]

        # Have each plot's corresp. axes update its tick spacing with value
        for pltNum in pltIndices:
            plt = self.plotItems[pltNum]
            axis = plt.getAxis(name)
            axis.setCstmTickSpacing(value)

        self.additionalUpdates()

    # Placeholder function to update plot ticks elsewhere as needed
    def additionalUpdates(self):
        pass

class MagPyTickIntervals(TickIntervals):
    def __init__(self, window, plotItems, parent=None):
        links = window.lastPlotLinks
        TickIntervals.__init__(self, window, plotItems, parent, links=links)

    def additionalUpdates(self):
        # Update plot ranges so tick marks are uniform across axes/plots
        self.window.updateXRange()
        self.window.updateYRange()