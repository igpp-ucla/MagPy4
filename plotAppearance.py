from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
import functools

class PlotAppearanceUI(object):
    def setupUI(self, Frame, window, plotsInfo):
        Frame.setWindowTitle('Plot Appearance')
        Frame.resize(300, 200)
        layout = QtWidgets.QGridLayout(Frame)

        # Font size label setup
        self.titleSzLbl = QtWidgets.QLabel('Title font size: ')
        self.axisLblSzLbl = QtWidgets.QLabel('Axis label font size: ')

        # Title and axis spinboxes setup
        self.titleSzBox = QtWidgets.QSpinBox()
        self.titleSzBox.setMinimum(5)
        self.titleSzBox.setMaximum(30)

        self.axisLblSzBox = QtWidgets.QSpinBox()
        self.axisLblSzBox.setMinimum(5)
        self.axisLblSzBox.setMaximum(25)

        layout.addWidget(self.titleSzLbl, 0, 0, 1, 1)
        layout.addWidget(self.axisLblSzLbl, 1, 0, 1, 1)

        layout.addWidget(self.titleSzBox, 0, 1, 1, 1)
        layout.addWidget(self.axisLblSzBox, 1, 1, 1, 1)

        # Set up UI for setting plot trace colors, line style, thickness, etc.
        tracePropFrame = QtWidgets.QGroupBox('Line Properties')
        tracePropFrame.setAlignment(QtCore.Qt.AlignCenter)
        tracePropLayout = QtWidgets.QVBoxLayout(tracePropFrame)

        pltNum = 0
        self.lineWidthBoxes = []
        self.lineStyleBoxes = []
        self.colorBoxes = []

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

            tracePropLayout.addWidget(plotFrame)
            pltNum += 1

        layout.addWidget(tracePropFrame, 3, 0, 1, 2)

class PlotAppearance(QtGui.QFrame, PlotAppearanceUI):
    def __init__(self, window, plotItems, parent=None):
        super(PlotAppearance, self).__init__(parent)
        self.ui = PlotAppearanceUI()
        self.plotItems = plotItems
        self.window = window

        # Get plots' trace/label infos and use to setup/initialize UI elements
        plotsInfo = self.getPlotsInfo()
        self.ui.setupUI(self, window, plotsInfo)
        self.initVars(plotsInfo)

        # Connect buttons to functions
        self.ui.titleSzBox.valueChanged.connect(self.changeTitleSize)
        self.ui.axisLblSzBox.valueChanged.connect(self.changeAxisLblSize)

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

    def setButtonColor(self, cs, color):
        styleSheet = "* { background:" + color.name() + " }"
        cs.setStyleSheet(styleSheet)
        cs.show()

    # Placeholder func to be implemented if title colors must match line colors
    def adjustTitleColors(self, penList):
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

class MagPyPlotApp(PlotAppearance):
    def __init__(self, window, plotItems, parent=None):
        PlotAppearance.__init__(self, window, plotItems, parent)

    def adjustTitleColors(self, penList):
        self.window.pltGrd.adjustTitleColors(penList)
        self.window.plotTracePens = penList

class SpectraPlotApp(PlotAppearance):
    def __init__(self, window, plotItems, parent=None):
        PlotAppearance.__init__(self, window, plotItems, parent)

    def adjustTitleColors(self, penList):
        self.window.updateTitleColors(penList)
        self.window.tracePenList = penList
