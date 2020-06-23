from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from .MagPy4UI import StackedLabel
from .plotBase import DateAxis

from datetime import datetime, timedelta
import numpy as np
from .layoutTools import BaseLayout

import pyqtgraph as pg
from bisect import bisect
import functools

class PlotAppearanceUI(BaseLayout):
    def setupUI(self, Frame, window, plotsInfo, plotItems, 
        mainWindow=False, links=None):
        Frame.setWindowTitle('Plot Appearance')
        Frame.resize(300, 200)

        # Set up tab widget in layout
        layout = QtWidgets.QGridLayout(Frame)
        self.layout = layout
        tw = QtWidgets.QTabWidget()
        self.tabWidget = tw
        layout.addWidget(tw, 3, 0, 1, 4)

        # Set up UI for setting plot trace colors, line style, thickness, etc.
        tracePropFrame = self.getTracePropFrame(plotsInfo)

        pltNum = 0 # Lists for storing interactive UI elements
        self.lineWidthBoxes = []
        self.lineStyleBoxes = []
        self.colorBoxes = []

        # If there are a lot of traces, wrap the trace properties frame
        # inside a scroll area
        tracePropScroll = self.wrapTracePropFrame(tracePropFrame)
        tw.addTab(tracePropScroll, 'Trace Properties')

        # Set up tick intervals widget
        if mainWindow:
            tickIntWidget = MagPyTickIntervals(window, plotItems, Frame)
        else:
            tickIntWidget = TickIntervals(window, plotItems, Frame, links=links)
        tw.addTab(tickIntWidget, 'Tick Spacing')

        # Set up label properties widget
        self.lblPropWidget = LabelAppear(window, plotItems, mainWindow)
        tw.addTab(self.lblPropWidget, 'Label Properties')
    
    def getTracePropFrame(self, plotInfos):
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(frame)
        self.styleGrps, self.widthGrps, self.colorGrps = [], [], []
        for index, info in plotInfos:
            plotFrame, elems = self.getPlotPropFrame(index, len(info))

            # Hide if special plot
            if len(info) == 0:
                plotFrame.setVisible(False)

            styleBoxes, widthBoxes, colorBtns = elems
            self.styleGrps.append(styleBoxes)
            self.widthGrps.append(widthBoxes)
            self.colorGrps.append(colorBtns)
            layout.addWidget(plotFrame)
        layout.addStretch()
        
        return frame

    def getPlotPropFrame(self, index, numTraces):
        # Build the properties frame for a single plot item
        frame = QtWidgets.QGroupBox(f'Plot {index+1}:')
        layout = QtWidgets.QGridLayout(frame)
        layout.setHorizontalSpacing(12)

        # Build header layout
        labels = [f'{label}:' for label in ['#', 'Style', 'Width', 'Color']]
        for i, label in enumerate(labels):
            layout.addWidget(QtWidgets.QLabel(label), 0, i, 1, 1)

        # Create line properties settings layout for each line
        styleBoxes, widthBoxes, colorBtns = [], [], []
        for index in range(0, numTraces):
            elems = self.getLinePropLt(index)
            for i, elem in enumerate(elems):
                layout.addWidget(elem, index+1, i, 1, 1)

            # Store line property elements
            lbl, style, width, color = elems
            styleBoxes.append(style)
            widthBoxes.append(width)
            colorBtns.append(color)

        return frame, (styleBoxes, widthBoxes, colorBtns)

    def getLinePropLt(self, index):
        # Line index label
        label = QtWidgets.QLabel(f'{index+1}:')
        label.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        # Create all elements for choosing line style
        lineStyle = QtWidgets.QComboBox()
        for t in ['Solid', 'Dashed', 'Dotted', 'DashDot']:
            lineStyle.addItem(t)
        
        # Create all elements for choosing line thickness
        lineWidth = QtWidgets.QDoubleSpinBox()
        lineWidth.setMinimum(1)
        lineWidth.setMaximum(5)
        lineWidth.setSingleStep(0.5)

        # Create elements for choosing line color
        colorBtn = QtWidgets.QPushButton()
        colorBtn.setCursor(QtCore.Qt.ClosedHandCursor)

        return (label, lineStyle, lineWidth, colorBtn)

    def wrapTracePropFrame(self, widgetFrame):
        # Create a scroll area and set its dimensions
        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(widgetFrame)
        scrollArea.setWidgetResizable(True)
        scrollArea.setMinimumHeight(min(widgetFrame.sizeHint().height(), 500))
        scrollArea.setMinimumWidth(widgetFrame.sizeHint().width()+150)
        scrollArea.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        # Set style properties to match tab widget's background
        scrollArea.setStyleSheet('QScrollArea {background-color: transparent;}')
        widgetFrame.setStyleSheet('QFrame {background-color: white;}')

        # Hide bottom scroll bar
        scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        return scrollArea

class PlotAppearance(QtGui.QFrame, PlotAppearanceUI):
    def __init__(self, window, plotItems, parent=None, mainWindow=False,
        links=None):
        super(PlotAppearance, self).__init__(parent)
        self.ui = PlotAppearanceUI()
        self.plotItems = plotItems
        self.window = window

        # Get plots' trace/label infos and use to setup/initialize UI elements
        plotsInfo = self.getPlotLineInfos()
        self.ui.setupUI(self, window, plotsInfo, plotItems, mainWindow, links=links)
        self.initVars(plotsInfo)

        # Connect line width modifiers to function
        grps = (plotsInfo, self.ui.styleGrps, self.ui.widthGrps, self.ui.colorGrps)
        for (index, info), styleGrp, widthGrp, colorGrp in zip(*grps):
            traceIndex = 0
            for line, style, width, color in zip(info, styleGrp, widthGrp, colorGrp):
                line['plotIndex'] = index
                line['traceIndex'] = traceIndex
                widthFunc = functools.partial(self.updateLineWidth, line)
                width.valueChanged.connect(widthFunc)

                styleFunc = functools.partial(self.updateLineStyle, line)
                style.currentTextChanged.connect(styleFunc)

                colorFunc = functools.partial(self.openColorSelect, color, line)
                color.clicked.connect(colorFunc)
                traceIndex += 1
    
        self.plotInfo = plotsInfo

    def getPlotLineInfos(self):
        # Creates list of per-plot lists containing tuples of pens and a list of
        # data items within the given plot that correspond to it
        # ex: [ [(pen1, [plotDataItem1...]), (pen2, [..])] , ... , ]
        n = len(self.plotItems)
        plotInfos = [(i, plt.getLineInfo()) for i, plt in zip(range(n), self.plotItems)]
        return plotInfos

    def getPenStyle(self, pen):
        # Extract style information from pen
        style = pen.style()
        width = pen.widthF()
        color = pen.color()

        return (style, width, color)
    
    def initVars(self, plotsInfo):
        # Initializes all values in UI elements with current plot properties
        if self.plotItems == None:
            return

        # Maps Qt Style flag to string
        styleMap = {
            QtCore.Qt.DashLine: 'Dashed',
            QtCore.Qt.DotLine: 'Dotted',
            QtCore.Qt.DashDotLine:'DashDot',
        }

        # Iterate over plots and their traces to initialize pen settings
        for index, info in plotsInfo:
            lineNum = 0
            for lineInfo in info:
                pen = lineInfo['pen']
                style, width, color = self.getPenStyle(pen)

                # Map the Qt Pen style to a string
                if style in styleMap:
                    styleStr = styleMap[style]
                else:
                    styleStr = 'Solid'
                
                # Set UI elements with values
                self.ui.styleGrps[index][lineNum].setCurrentText(styleStr)
                self.ui.widthGrps[index][lineNum].setValue(width)
                self.setButtonColor(self.ui.colorGrps[index][lineNum], color)

                lineNum += 1

    def updateLineWidth(self, lineInfo, val):
        # Extract line and its pen item
        line = lineInfo['item']
        pen = lineInfo['pen']

        # Set new pen width
        pen.setWidthF(val)
        line.setPen(pen)
        lineInfo['pen'] = pen

        # Apply any other persistent changes TODO:
        self.setChangesPersistent(lineInfo)

    def updateLineStyle(self, lineInfo, styleStr):
        # Get style object from name
        if styleStr == 'Dashed':
            style = QtCore.Qt.DashLine
        elif styleStr == 'Dotted':
            style = QtCore.Qt.DotLine
        elif styleStr == 'DashDot':
            style = QtCore.Qt.DashDotLine
        else:
            style = QtCore.Qt.SolidLine

        # Update pens for selected plots
        line = lineInfo['item']
        pen = lineInfo['pen']
        pen.setStyle(style)
        line.setPen(pen)
        lineInfo['pen'] = pen

        # TODO: Apply other persistent changes
        self.setChangesPersistent(lineInfo)

    def openColorSelect(self, btn, lineInfo):
        # Open color selection dialog and connect to line color update function
        clrDialog = QtWidgets.QColorDialog(self)
        clrDialog.show()
        clrDialog.setCurrentColor(lineInfo['pen'].color())
        clrDialog.colorSelected.connect(functools.partial(self.setLineColor, btn, lineInfo))

    def setLineColor(self, btn, lineInfo, color):
        # Update pen color of every trace item
        pen = lineInfo['pen']
        pen.setColor(color)
        lineInfo['pen'] = pen
        
        line = lineInfo['item']
        line.setPen(pen)

        # Match the color selection button to the selected color
        self.setButtonColor(btn, color)

        # Set the title colors to match, if implemented
        self.setChangesPersistent(lineInfo)
        self.adjustTitleColors(lineInfo)

    def setButtonColor(self, cs, color):
        styleSheet = f'''
            * {{
                background: {color.name()};
            }}
        '''
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

    def closeEvent(self, event):
        if self.ui.lblPropWidget.textEditor:
            self.ui.lblPropWidget.closeLabelEditor()
        self.close()

    def allowLeftAxisEditing(self):
        # Allow left axis editing if not applying change to a color plot
        return True

class MagPyPlotApp(PlotAppearance):
    def __init__(self, window, plotItems, parent=None):
        PlotAppearance.__init__(self, window, plotItems, parent, True)

    def adjustTitleColors(self, lineInfo):
        ''' Adjusts StackedLabel colors and pens stored in state '''
        plotIndex = lineInfo['plotIndex']
        lineIndex = lineInfo['traceIndex']
        pen = lineInfo['pen']

        # Set colors in PlotGrid labels
        self.window.pltGrd.adjustTitleColors(lineInfo)

    def setChangesPersistent(self, lineInfo):
        # Set pen in window state
        plotIndex = lineInfo['plotIndex']
        lineIndex = lineInfo['traceIndex']
        pen = lineInfo['pen']
        self.window.plotTracePens[plotIndex][lineIndex] = pen

class SpectraPlotApp(PlotAppearance):
    def __init__(self, window, plotItems, parent=None):
        PlotAppearance.__init__(self, window, plotItems, parent)

    def adjustTitleColors(self, lineInfo):
        self.window.updateTitleColors()

    def setChangesPersistent(self, lineInfo):
        plotIndex = lineInfo['plotIndex']
        lineIndex = lineInfo['traceIndex']
        pen = lineInfo['pen']
        self.window.tracePenList[plotIndex][lineIndex] = pen

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

class DynamicPlotApp(PlotAppearance):
    def __init__(self, window, plotItems, parent=None):
        PlotAppearance.__init__(self, window, plotItems, parent)
        self.ui.tabWidget.removeTab(0)
        self.ui.tabWidget.removeTab(1)

    def allowLeftAxisEditing(self):
        return True

class PressurePlotApp(PlotAppearance):
    def adjustTitleColors(self, penList):
        penIndex = 0
        plotIndex = 0
        for penGrp in penList:
            row = 0
            for pen in penGrp:
                for j in range(row, row+2):
                    sub_label = self.window.labels[plotIndex].subLabels[j]
                    fontSize = sub_label.opts['size']
                    sub_label.setText(sub_label.text, size=fontSize, color=pen.color().name())
                row += 2
                self.window.pens[penIndex] = pen
                penIndex += 1
            plotIndex += 1

class TickIntervalsUI(BaseLayout):
    def setupUI(self, Frame, window, plotItems, links):
        Frame.setWindowTitle('Set Tick Spacing')
        Frame.resize(200, 100)
        layout = QtWidgets.QGridLayout(Frame)

        btmPlt = plotItems[-1]

        # Determine Y val with largest magnitude to set spinbox max for left axes
        yMax = 100
        for plt in plotItems:
            if plt.isSpecialPlot():
                continue
            for pdi in plt.listDataItems():
                yMax = max(yMax, max(np.absolute(pdi.yData)))

        # Collect UI boxes/buttons and information about the corresp. axis
        self.intervalBoxes = []
        self.timeLblBoxes = []
        for name in ['left', 'bottom']:
            if name == 'left' and not Frame.Frame.allowLeftAxisEditing():
                continue
            # Builds frames / UI elements for corresponding axis side
            self.buildAxisBoxes(layout, name, plotItems, yMax, links)
        spacer = QtWidgets.QSpacerItem(0, 0, QSizePolicy.Maximum, QSizePolicy.MinimumExpanding)
        layout.addItem(spacer, layout.count(), 0, 1, 1)

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
            # Set label format if it is not set to default
            if ax.labelFormat is not None:
                for tickBox, labelBox in self.timeLblBoxes:
                    if tickBox == box:
                        # Get index associated w/ format
                        itemIndex = 0
                        items = [labelBox.itemText(i) for i in range(labelBox.count())]
                        for item in items:
                            if item == ax.labelFormat:
                                labelBox.setCurrentIndex(itemIndex)
                                break
                            itemIndex += 1
        else:
            # Set initial value for spinbox if previously set
            if ax.tickDiff is not None:
                box.setValue(ax.tickDiff)
            else:
                box.setValue(1)

    def buildTimeLblLt(self):
        # Get time label formats
        fmts = DateAxis(orientation='bottom', epoch='J2000').timeModes
        fmts = ['Default'] + fmts
        # Build box and layout elements
        layout = QtWidgets.QGridLayout()
        fmtBox = QtWidgets.QComboBox()
        fmtBox.addItems(fmts)
        self.addPair(layout, 'Label Format: ', fmtBox, 0, 0, 1, 1)
        # Add in a spacer
        spacer = QtWidgets.QSpacerItem(0, 0, QSizePolicy.MinimumExpanding, QSizePolicy.Maximum)
        layout.addItem(spacer, 0, 2, 1, 1)
        return layout, fmtBox

    def buildAxisBoxes(self, layout, name, plotItems, yMax, links):
        # For every linked axis group
        numLinks = len(links) if (name == 'left' and links is not None) else 1
        for i in range(0, numLinks):
            # Get axis, axis type, and create UI elements as apprp.
            if links:
                linkSubset = links[i]
                btmPlt = plotItems[linkSubset[-1]]
            else:
                btmPlt = plotItems[-1]
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
            layout.addWidget(axisFrame, layout.count(), 0)

    def buildIntervalUI(self, name, axType):
        # Initialize common UI elements
        axisFrame = QtWidgets.QGroupBox(name + ':')
        applyBtn = QtWidgets.QPushButton('Apply')
        defBtn = QtWidgets.QPushButton('Default')

        if axType == 'DateTime':
            # Create layout w/ timeEdit and label to describe time sections
            dateTimeLt = QtWidgets.QVBoxLayout(axisFrame)
            lbl = QtWidgets.QLabel('HH:MM:SS')
            intervalBox = QtWidgets.QTimeEdit()
            intervalBox.setDisplayFormat('HH:mm:ss')
            intervalBox.setMinimumTime(QtCore.QTime(0, 0, 1))

            for rowLst in [[lbl], [intervalBox, defBtn, applyBtn]]:
                hlt = QtWidgets.QHBoxLayout()
                for elem in rowLst:
                    hlt.addWidget(elem, QtCore.Qt.AlignLeft)
                spacer = QtWidgets.QSpacerItem(0, 0, QSizePolicy.MinimumExpanding)
                hlt.addItem(spacer)
                dateTimeLt.addLayout(hlt)
            
            timestampLt, fmtBox = self.buildTimeLblLt()
            dateTimeLt.addLayout(timestampLt)
            self.timeLblBoxes.append((intervalBox, fmtBox))
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
            intervalLt.addWidget(intervalBox, QtCore.Qt.AlignLeft)
            intervalLt.addWidget(defBtn, QtCore.Qt.AlignLeft)
            intervalLt.addWidget(applyBtn, QtCore.Qt.AlignLeft)
            spacer = QtWidgets.QSpacerItem(0, 0, QSizePolicy.MinimumExpanding)
            intervalLt.addItem(spacer)

        # Set maximums to adjust for resizing
        for elem in [intervalBox, applyBtn, defBtn]:
            elem.setMaximumWidth(150)

        return axisFrame, intervalBox, applyBtn, defBtn

class TickIntervals(QtGui.QFrame, TickIntervalsUI):
    def __init__(self, window, plotItems, Frame, parent=None, links=None):
        self.Frame = Frame
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

            # Connect date time axis label format box to appropriate axes
            for refBox, lblBox in self.ui.timeLblBoxes:
                if refBox == intBox:
                    lblBox.currentIndexChanged.connect(functools.partial(self.updateLabels, lblBox, name, i))
                    break

    def updateLabels(self, box, name, index):
        # Set date time axis labels format
        pltIndices = self.getPlotIndices(name, index)
        fmtKw = box.currentText()

        for pltNum in pltIndices:
            plt = self.plotItems[pltNum]
            axis = plt.getAxis(name)
            if fmtKw == 'Default':
                axis.resetLabelFormat()
            elif fmtKw is not None:
                axis.setLabelFormat(fmtKw)
            self.adjustOpposingAxis(plt, name)

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
            self.adjustOpposingAxis(plt, name)

        self.additionalUpdates(None, name)

    def adjustOpposingAxis(self, plt, axName):
        pairs = [('left', 'right'), ('top', 'bottom')]

        for a, b in pairs:
            if axName == a:
                otherAx = b
            elif axName == b:
                otherAx = a
        
        refAxis = plt.getAxis(axName)
        otherAxis = plt.getAxis(otherAx)

        if otherAxis.style['tickLength'] == 0 or (not otherAxis.isVisible()):
            return

        # Set custom tick spacing
        tickDiff = refAxis.tickDiff
        otherAxis.setCstmTickSpacing(tickDiff)

        # Set label format if a datetime axis
        if refAxis.axisType() == 'DateTime':
            label = refAxis.labelFormat
            otherAxis.labelFormat = label

        otherAxis.picture = None
        otherAxis.update()

    def getPlotIndices(self, name, index):
        if self.links is not None and name == 'left':
            pltIndices = self.links[index]
        else:
            pltIndices = [pltNum for pltNum in range(len(self.plotItems))]
        return pltIndices

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
        pltIndices = self.getPlotIndices(name, index)

        # Have each plot's corresp. axes update its tick spacing with value
        for pltNum in pltIndices:
            plt = self.plotItems[pltNum]
            axis = plt.getAxis(name)
            axis.setCstmTickSpacing(value)
            self.adjustOpposingAxis(plt, name)

        self.additionalUpdates(value, name)

    # Placeholder function to update plot ticks elsewhere as needed
    def additionalUpdates(self, tickDiff, name):
        pass

class MagPyTickIntervals(TickIntervals):
    def __init__(self, window, plotItems, parent=None):
        links = window.lastPlotLinks
        TickIntervals.__init__(self, window, plotItems, parent, links=links)

    def additionalUpdates(self, tickDiff, name):
        # Update plot ranges so tick marks are uniform across axes/plots
        if name == 'bottom' and self.window.pltGrd and self.window.pltGrd.labelSetGrd:
            self.window.pltGrd.labelSetGrd.setCstmTickSpacing(tickDiff)
        self.window.updateXRange()
        self.window.updateYRange()

class LabelAppearUI(BaseLayout):
    def setupUI(self, Frame, window):
        frameLt = QtWidgets.QVBoxLayout(Frame)
        layout = QtWidgets.QGridLayout()
        self.layout = layout

        # Font size label setup
        self.titleSzLbl = QtWidgets.QLabel('Title Size: ')
        self.axisLblSzLbl = QtWidgets.QLabel('Axis Label Size: ')
        self.tickLblSzLbl = QtWidgets.QLabel('Tick Label Size: ')

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

        lbls = [self.titleSzLbl, self.axisLblSzLbl, self.tickLblSzLbl]
        boxes = [self.titleSzBox, self.axisLblSzBox, self.tickLblSzBox]
        row = 0
        for lbl, box in zip(lbls, boxes):
            layout.addWidget(lbl, row, 0, 1, 1, QtCore.Qt.AlignLeft)
            layout.addWidget(box, row, 1, 1, 1, QtCore.Qt.AlignLeft)
            lbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))
            box.setFixedWidth(100)
            # Fill in empty space to right
            spacer = QtWidgets.QSpacerItem(0, 0, QSizePolicy.MinimumExpanding)
            layout.addItem(spacer, row, 2, 1, 1)
            row += 1

        # Add in default box for axis label sizes in main window
        spacer = layout.itemAtPosition(1, 2)
        layout.removeItem(spacer)
        self.defaultBtn = QtWidgets.QPushButton('Default')
        self.defaultBtn.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        layout.addWidget(self.defaultBtn, 1, 2, 1, 1)

        frameLt.addLayout(layout)
        self.labelEditorBtn = QtWidgets.QPushButton('Edit Label Text...')
        layout.addWidget(self.labelEditorBtn, row, 0, 1, 2)
        frameLt.addStretch()

class LabelAppear(QtWidgets.QFrame, LabelAppearUI):
    def __init__(self, window, plotItems, inMainWindow=False, parent=None):
        super(LabelAppear, self).__init__(parent)
        self.ui = LabelAppearUI()
        self.plotItems = plotItems
        self.window = window
        self.textEditor = None
        self.inMainWindow = inMainWindow

        # Get plots' trace/label infos and use to setup/initialize UI elements
        self.ui.setupUI(self, window)
        self.initVars()

        # Connect spinbox changes to functions
        self.ui.titleSzBox.valueChanged.connect(self.changeTitleSize)
        self.ui.axisLblSzBox.valueChanged.connect(self.changeAxisLblSize)
        self.ui.tickLblSzBox.valueChanged.connect(self.changeTickLblSize)
        self.ui.defaultBtn.clicked.connect(self.resizeMainAxisLbls)

        if inMainWindow:
            self.ui.labelEditorBtn.clicked.connect(self.openLabelEditor)
        else: # Remove label editor button for non-main-window menus
            self.ui.layout.removeWidget(self.ui.labelEditorBtn)
            self.ui.labelEditorBtn.deleteLater()
            self.ui.defaultBtn.setVisible(False)

    def openLabelEditor(self):
        self.closeLabelEditor()
        self.textEditor = RenameLabels(self.window)
        self.textEditor.show()
        self.textEditor.initVars()

    def resizeMainAxisLbls(self):
        # Unlock label resize settings and resize plot grid so they
        # are reverted to default sizes
        self.window.pltGrd.lockLabelSizes(False)
        self.window.pltGrd.resizeEvent(None)

    def closeLabelEditor(self):
        if self.textEditor:
            self.textEditor.close()

    def initVars(self):
        plt = self.plotItems[-1]
        # Get title font size to initialize spin box, disable if no title
        titleSize = plt.titleLabel.opts['size'][:-2] # Strip pts part of string
        if plt.titleLabel.text == '':
            for elem in [self.ui.titleSzBox, self.ui.titleSzLbl]:
                self.ui.layout.removeWidget(elem)
                elem.deleteLater()
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
        if axis.getTickFont() == None: # No custom font, use default
            self.ui.tickLblSzBox.setValue(11)
        else:
            tickFont = axis.getTickFont()
            tickSize = tickFont.pointSize()
            if tickSize < 0: # No point size set, use default
                tickSize = 11
            self.ui.tickLblSzBox.setValue(int(tickSize))
        self.ui.tickLblSzBox.blockSignals(False)

    def changeTitleSize(self, val):
        for plt in self.plotItems:
            plt.titleLabel.setText(plt.titleLabel.text, size=str(val)+'pt')

    def changeAxisLblSize(self, val):
        # Set stacked label sizes for main window plot grid and lock label size
        if self.inMainWindow and self.window.pltGrd:
            self.window.pltGrd.setLabelFontSizes(val)
            self.window.pltGrd.lockLabelSizes()
            return

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
                axis.setTickFont(tickFont)

                # Adjust vert/horz spacing reserved for bottom ticks if necessary
                self.adjustTickHeights(axis, tickFont)

        if self.inMainWindow and self.window.pltGrd:
            if self.window.pltGrd.labelSetGrd:
                self.window.pltGrd.setLabelSetFontSize(val)
            ax = self.plotItems[-1].getAxis('bottom')
            ax.labelStyle = {'font-size':f'{val}pt'}
            self.window.pltGrd.adjustPlotWidths()

    def adjustTickHeights(self, axis, tickFont):
        # Adjust vertical spacing reserved for bottom ticks if necessary
        mets = QtGui.QFontMetrics(tickFont)
        ht = mets.boundingRect('AJOW').height() # Tall letters
        if ht > 18 and axis.orientation == 'bottom':
            ht = max(int(ht*.25), 5)
            axis.setStyle(tickTextOffset=ht)
        elif axis.orientation == 'bottom':
            axis.setStyle(tickTextOffset=2)

class RenameLabelsUI(BaseLayout):
    def setupUI(self, Frame, window):
        self.Frame = Frame
        Frame.setWindowTitle('Label Editor')
        Frame.setStyleSheet('QFrame { background-color: white; } \
            QScrollArea { background-color : white; }')
        layout = QtWidgets.QVBoxLayout(Frame)
        self.emptyKw = '_Empty_'

        self.tables = [] # Text list widgets
        self.colorTables = [] # Corresponding color list widgets

        # Build label editor frame for each stacked label in plot grid
        plotNum = 0
        plotFrames = []
        for lbl in window.pltGrd.labels:
            plotFrame = QtWidgets.QGroupBox('Plot '+str(plotNum+1) +':')
            plotLayout = QtWidgets.QGridLayout(plotFrame)

            # Set up tables
            tableLt, colorTable, columnWidget = self.getTables()
            self.tables.append(columnWidget)
            self.colorTables.append(colorTable)

            # Set up add/remove buttons
            addBtn = QtWidgets.QPushButton('+')
            rmvBtn = QtWidgets.QPushButton('−')

            plotLayout.addLayout(tableLt, 0, 0, 2, 1)
            plotLayout.addWidget(addBtn, 0, 1, 1, 1)
            plotLayout.addWidget(rmvBtn, 1, 1, 1, 1)

            # Connect actions to signals
            rmvBtn.clicked.connect(functools.partial(self.rmvFrmLst, plotNum))
            addBtn.clicked.connect(functools.partial(self.addToLst, plotNum))

            for btn in [rmvBtn, addBtn]:
                btn.setFixedWidth(50)

            plotFrames.append(plotFrame)
            plotNum += 1

        # Wrap editor sublayouts in a scroll area if there are too many labels
        if plotNum > 5:
            scrollFrame = QtWidgets.QFrame()
            innerLayout = QtWidgets.QVBoxLayout(scrollFrame)

            for frm in plotFrames: # Add in all sub frames
                innerLayout.addWidget(frm)

            scrollArea = QtWidgets.QScrollArea()
            scrollArea.setWidget(scrollFrame)
            scrollArea.setMinimumWidth(400)
            scrollArea.setMinimumHeight(450)
            scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

            layout.addWidget(scrollArea)
        else: # Add widget to frame directly
            for frm in plotFrames:
                layout.addWidget(frm)

        # Apply/Default buttons setup
        applyLt = QtWidgets.QHBoxLayout()
        self.applyBtn = QtWidgets.QPushButton('Apply')
        self.defaultBtn = QtWidgets.QPushButton('Defaults')
        applyLt.addWidget(self.defaultBtn)
        applyLt.addWidget(self.applyBtn)
        layout.addLayout(applyLt)

    def colorToData(self, color):
        return QtCore.QVariant(color)

    def iconFromColor(self, color):
        pix = QtGui.QPixmap(12, 12)
        pix.fill(pg.mkColor(color))
        icon = QtGui.QIcon(pix)
        return icon

    def itemFromColor(self, color):
        # Creates a listWidgetItem w/ a colored icon from a hex color string
        icon = self.iconFromColor(color)
        item = QtWidgets.QListWidgetItem(icon, '')
        return item

    def scrollTable(self, table, val):
        # Scroll a list widget without signals
        table.blockSignals(True)
        table.verticalScrollBar().setValue(val)
        table.blockSignals(False)

    def getTables(self):
        # Set up color table and dstr table in a single layout
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setHorizontalSpacing(1)

        # Create the text list widget
        dstrTable = QtWidgets.QListWidget()
        font = QtGui.QFont('12pt')
        dstrTable.setFont(font)
        dstrTable.setStyleSheet('QListWidget::item { border-bottom: 1px solid #bfbfbf; \
            padding-bottom: 2px ; padding-top: 2px}')

        # Set up color table w/ a fixed width and connect to color select
        colorTable = QtWidgets.QListWidget()
        colorTable.setMaximumWidth(20)
        colorTable.setMinimumWidth(20)
        colorTable.itemClicked.connect(self.openColorSelect)
        colorTable.setStyleSheet('QListWidget::item { border: 0px solid black; padding-bottom: 2px ; padding-top: 2px }')

        # Disable selection highlighting and hide scrollbars in colors list
        colorTable.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        colorTable.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        colorTable.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # Link scrolling between tables
        func = colorTable.verticalScrollBar().setValue
        dstrTable.verticalScrollBar().valueChanged.connect(func)
        func = functools.partial(self.scrollTable, dstrTable)
        colorTable.verticalScrollBar().valueChanged.connect(func)

        layout.addWidget(colorTable, 0, 0, 1, 1)
        layout.addWidget(dstrTable, 0, 1, 1, 1)

        return layout, colorTable, dstrTable

    def rmvFrmLst(self, plotNum):
        # Removes the bottom text/color from each list for the given plot number
        if len(self.tables) == 0:
            return
        table = self.tables[plotNum]
        colorTable = self.colorTables[plotNum]
        row = table.count() - 1
        table.takeItem(row)
        colorTable.takeItem(row)

    def addToLst(self, plotNum):
        # Adds an 'empty' element for each list for the given plot
        table = self.tables[plotNum]
        table.addItem(self.emptyKw)
        colorTable = self.colorTables[plotNum]
        colorTable.addItem(self.itemFromColor('#000000'))
        # Get last item in list (just added) and set editable
        item = table.item(table.count()-1)
        table.openPersistentEditor(item)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

    def openColorSelect(self, item):
        # Open color selection dialog and connect to color update function
        clrDialog = QtWidgets.QColorDialog(self.Frame)
        clrDialog.show()
        func = functools.partial(self.setColorForItem, item)
        clrDialog.colorSelected.connect(func)

    def setColorForItem(self, item, color):
        # Sets the color of a list widget item
        icon = self.iconFromColor(color)
        item.setIcon(icon)
        item.setData(QtCore.Qt.UserRole, self.colorToData(color.name()))

    def getPlotInfo(self, plotNum):
        table = self.tables[plotNum]
        colorTable = self.colorTables[plotNum]

        dstrs, colors = [], []
        for row in range(0, table.count()):
            dstr = table.item(row).text()
            color = colorTable.item(row).data(QtCore.Qt.UserRole)
            if dstr != self.emptyKw:
                dstrs.append(dstr)
                colors.append(color)

        return dstrs, colors

    def clearPlot(self, plotNum):
        for table in [self.tables[plotNum], self.colorTables[plotNum]]:
            table.clear()

    def fillPlot(self, plotNum, dstrs, colors):
        self.clearPlot(plotNum)

        table = self.tables[plotNum]
        table.addItems(dstrs)
        for row in range(0, table.count()):
            item = table.item(row)
            table.openPersistentEditor(item)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

        colorTable = self.colorTables[plotNum]
        colorItems = list(map(self.itemFromColor, colors))
        for item, color in zip(colorItems, colors):
            item.setData(QtCore.Qt.UserRole, color)
            colorTable.addItem(item)

class RenameLabels(QtWidgets.QFrame, RenameLabelsUI):
    def __init__(self, window, parent=None):
        super(RenameLabels, self).__init__(parent)
        self.ui = RenameLabelsUI()
        self.window = window
        self.plotGrid = window.pltGrd
        self.defaultPlotStrings = window.lastPlotStrings
        self.defaultPens = window.plotTracePens

        self.ui.setupUI(self, window)
        self.ui.applyBtn.clicked.connect(self.rebuildLabels)
        self.ui.defaultBtn.clicked.connect(self.resetToDefaults)

    def initVars(self):
        plotNum = 0
        for lbl in self.plotGrid.labels:
            self.ui.fillPlot(plotNum, lbl.dstrs, lbl.colors)
            plotNum += 1

    def resetToDefaults(self):
        plotNum = 0
        for pltLst, penLst in zip(self.defaultPlotStrings, self.defaultPens):
            if penLst == None:
                # If a color plot, extract the default title and units from
                # the plot grid's state information
                index = self.defaultPlotStrings.index(pltLst)
                plt = self.plotGrid.plotItems[index]
                cpIndex = self.plotGrid.colorPlts.index(plt)
                name = self.plotGrid.colorPltNames[cpIndex]
                units = self.plotGrid.colorPltUnits[cpIndex]
                dstrs = [name, '['+units+']']
                colors = ['#000000', '#888888']
                self.ui.fillPlot(plotNum, dstrs, colors)
                plotNum += 1
                continue
            # Get default colors from pen list and use to initialize a stacked lbl
            colors = list(map(QtGui.QPen.color, penLst))
            stackedLbl = self.window.buildStackedLabel(pltLst, colors)
            dstrs = stackedLbl.dstrs
            colors = stackedLbl.colors
            self.ui.fillPlot(plotNum, dstrs, colors)
            plotNum += 1

    def rebuildLabels(self):
        # Create new labels and replace labels in plot grid with them
        for plotNum in range(0, len(self.window.plotItems)):
            prevLbl = self.plotGrid.labels[plotNum]
            prevDstrs, prevColors = prevLbl.dstrs, prevLbl.colors

            newDstrs, newColors = self.ui.getPlotInfo(plotNum)
            lbl = StackedLabel(newDstrs, newColors)
            self.plotGrid.setPlotLabel(lbl, plotNum)