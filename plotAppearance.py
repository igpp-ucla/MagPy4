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
                styleLabel = QtWidgets.QLabel('Style: ')
                lineStyle = QtWidgets.QComboBox()
                for t in ['Solid', 'Dashed', 'Dotted', 'DashDot']:
                    lineStyle.addItem(t)
                self.lineStyleBoxes.append((lineStyle, (pltNum, traceNum)))

                # Create all elements for choosing line thickness
                widthLabel = QtWidgets.QLabel('Width: ')
                lineWidth = QtWidgets.QSpinBox()
                lineWidth.setMinimum(1)
                lineWidth.setMaximum(5)
                self.lineWidthBoxes.append((lineWidth, (pltNum, traceNum)))

                # Add all elements to sublayout
                for e in [label, styleLabel, lineStyle, widthLabel, lineWidth]:
                    traceLayout.addWidget(e)
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

    def getPlotsInfo(self):
        # Creates list of per-plot lists containing tuples of pens and a list of
        # data items within the given plot that correspond to it
        # ex: [ [(pen1, [plotDataItem1...]), (pen2, [..])] , ... , ]
        plotsInfo = []
        for plt in self.plotItems:
            pltInfo = []
            uniqPltPens = []
            for pt in plt.listDataItems():
                pen = pt.opts['pen']
                if pen not in uniqPltPens:
                    uniqPltPens.append(pen)
                    pltInfo.append((pen, [pt]))
                else:
                    # Find corresp. pen in current plot info list
                    index = uniqPltPens.index(pen)
                    penItemList = pltInfo[index][1]
                    penItemList.append(pt)
            plotsInfo.append(pltInfo)
        return plotsInfo

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
            self.ui.axisLblSzBox.setValue(11)
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
                traceNum += 1

    def changeTitleSize(self, val):
        for plt in self.plotItems:
            plt.titleLabel.setText(plt.titleLabel.text, size=str(val)+'pt')

    def changeAxisLblSize(self, val):
        # Update every plot's label sizes for every axis
        for plt in self.plotItems:
            for ax in [plt.getAxis('bottom'), plt.getAxis('left')]:
                lblText = ax.label.toPlainText()
                if lblText == '': # Do not change size if no label
                    continue
                # Convert new value to apprp string and set
                sizeStr = str(val) + 'pt'
                labelStyle = {'font-size':sizeStr}
                ax.setLabel(lblText, **labelStyle)
