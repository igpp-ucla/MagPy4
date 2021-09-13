
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from ..qtinterface.layouttools import TimeEdit
from ..plotbase import GraphicsLayout, MagPyPlotItem
from ..qtinterface.plotuibase import GraphicsView
import functools

class RowGridLayout(GraphicsLayout):
    def __init__(self, maxCols=None, parent=None, border=None):
        self.maxCols = maxCols
        super().__init__()

    def clear(self):
        super().clear()
        self.currentRow = 0
        self.currentCol = 0

    def setNumCols(self, n):
        self.maxCols = n

    def addItem(self, item, row=None, col=None, rowspan=1, colspan=1):
        super().addItem(item, row, col, rowspan, colspan)

        # If current column (after item is placed) is >= maxCols,
        # move to the next row
        if self.maxCols is not None and self.currentCol >= self.maxCols:
            self.nextRow()

    def getRow(self, rowNum):
        # Get list of column numbers in sorted order and return items in row
        cols = list(self.rows[rowNum].keys())
        cols.sort()
        return [self.rows[rowNum][col] for col in cols]

    def getRowItems(self):
        ''' Returns list of items in each row as a list '''
        # Get list of row numbers in sorted order
        rowIndices = list(self.rows.keys())
        rowIndices.sort()

        # Get list of items in each row
        rowItems = []
        for row in rowIndices:
            rowItems.append(self.getRow(row))

        return rowItems

class SpectraGrid(RowGridLayout):
    def __init__(self, *args, **kwargs):
        self.name = ''
        self.window = None
        super().__init__(*args, **kwargs)

    def set_name(self, name):
        self.name = name
    
    def set_window(self, window):
        self.window = window

    def addItem(self, *args, **kwargs):
        super().addItem(*args, **kwargs)

        # Custom actions and helper windows
        self.plotAppr = None
        self.plotApprAction = QtWidgets.QAction(self)
        self.plotApprAction.triggered.connect(self.openPlotAppr)
        self.plotApprAction.setText('Change Plot Appearance...')
        self.updateRowWidths()

    def openPlotAppr(self):
        from .plotappr import PlotAppearance
        self.closePlotAppr()
        self.plotAppr = PlotAppearance(self)
        if self.window is not None:
            color_func = functools.partial(self.window.updateTitleColors, self.name)
            self.plotAppr.colorsChanged.connect(color_func)
        self.destroyed.connect(self.plotAppr.close)
        self.plotAppr.show()
    
    def closePlotAppr(self):
        if self.plotAppr:
            self.plotAppr.close()
            self.plotAppr = None

    def getContextMenus(self, *args, **kwargs):
        menus = [self.plotApprAction]
        return menus

    def updateRowWidths(self):
        rowItems = self.getRowItems()
        for row in rowItems:
            rowPlots = []
            minWidth = 10
            for item in row:
                if isinstance(item, pg.PlotItem):
                    rowPlots.append(item)
                    width = item.minimumWidth()
                    minWidth = max(width, minWidth)

            for plot in rowPlots:
                plot.setMinimumWidth(minWidth)
    
    def get_plots(self):
        items = self.getRowItems()
        plots = []
        for row in items:
            for item in row:
                if isinstance(item, MagPyPlotItem): 
                    plots.append(item)
        return plots

    def set_tick_text_size(self, val):
        plots = self.get_plots()
        font = QtGui.QFont()
        font.setPointSize(val)
        for plot in plots:
            for key in ['left', 'bottom', 'right', 'top']:
                ax = plot.getAxis(key)
                ax.setStyle(tickFont=font)

    def get_traces(self):
        traces = []
        plots = self.get_plots()
        index = 0
        for plot in plots:
            lines = plot.getLineInfo()
            traces.append((index, lines))
            index += 1
        return traces
    
    def get_links(self):
        plots = self.get_plots()
        links = [[i for i in range(len(plots))]]
        return links

    def update_label_color(self, plot, name, color):
        from .spectra import SpectraPlot
        if isinstance(plot, SpectraPlot):
            label = plot.getTitleObject()
            colors = label.getColors()
            labels = label.getLabels()
            index = -1
            for i in range(len(labels)):
                if labels[i] == name:
                    index = i

            if index >= 0:
                colors[index] = color
                label.setColors(colors)

class SpectraUI(object):
    def buildSpectraView(self):
        view = GraphicsView()
        view.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        gmain = pg.GraphicsLayout() # made this based off pg.GraphicsLayout
        gmain.setContentsMargins(11,0,11,0) # left top right bottom
        view.setCentralItem(gmain)
        grid = SpectraGrid(4)
        grid.setContentsMargins(0,0,0,0)
        labelLayout = pg.GraphicsLayout()
        labelLayout.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum))
        labelLayout.setContentsMargins(11, 3, 11, 5)
        gmain.addItem(grid, 0, 0, 1, 1)
        gmain.addItem(labelLayout, 1, 0, 1, 1)
        return view, grid, labelLayout

    def buildCombinedView(self):
        view = GraphicsView()
        view.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        gmain = pg.GraphicsLayout() # made this based off pg.GraphicsLayout
        #apparently default is 11, tried getting the margins and they all were zero seems bugged according to pyqtgraph
        gmain.setContentsMargins(11,0,11,11) # left top right bottom
        view.setCentralItem(gmain)
        grid = SpectraGrid(2)
        grid.setContentsMargins(0,0,0,0)
        gmain.addItem(grid)
        return view, grid

    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Spectra')
        Frame.resize(1000,700)

        self.plotApprAction = QtWidgets.QAction(window)
        self.plotApprAction.setText('Change Plot Appearance...')

        layout = QtWidgets.QVBoxLayout(Frame)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.gview, self.grid, self.labelLayout = self.buildSpectraView()
        self.tabs.addTab(self.gview, 'Spectra')

        self.cohView, self.cohGrid, self.cohLabelLayout = self.buildSpectraView()
        self.tabs.addTab(self.cohView, 'Coherence')

        self.phaView, self.phaGrid, self.phaLabelLayout = self.buildSpectraView()
        self.tabs.addTab(self.phaView, 'Phase')

        self.combView, self.sumGrid = self.buildCombinedView()
        self.tabs.addTab(self.combView, 'Sum of Powers')

        grids = [self.grid, self.phaGrid, self.cohGrid, self.sumGrid]
        labels = ['spectra', 'coherence', 'phase', 'sum']
        for grid, label in zip(grids, labels):
            grid.set_name(label)
            grid.set_window(Frame)

        # bandwidth label and spinbox
        bottomLayout = QtWidgets.QHBoxLayout()
        bandWidthLabel = QtWidgets.QLabel("Average Bandwidth:  ")
        self.bandWidthSpinBox = QtWidgets.QSpinBox()
        self.bandWidthSpinBox.setSingleStep(2)
        self.bandWidthSpinBox.setMinimum(1)
        self.bandWidthSpinBox.setProperty("value", 3)
        self.bandWidthSpinBox.setFixedWidth(50)

        # this will separate multiple traces on the same plot
        self.separateTracesCheckBox = QtWidgets.QCheckBox('Separate Traces')
        self.separateTracesCheckBox.setChecked(False)
        ###separateTraces = QtWidgets.QLabel("Separate Traces")
      
        self.aspectLockedCheckBox = QtWidgets.QCheckBox('Lock Aspect Ratio')
        self.aspectLockedCheckBox.setChecked(False)
        ###aspectLockedLabel = QtWidgets.QLabel("Lock Aspect Ratio")

        self.logModeCheckBox = QtWidgets.QCheckBox('Logarithmic scaling')
        self.logModeCheckBox.setChecked(True)

        self.unitRatioCheckbox = QtWidgets.QCheckBox('Unit Ratio')
        self.unitRatioCheckbox.setChecked(False)
        self.unitRatioCheckbox.setToolTip('Set X/Y axes to have same scale and be the same size')

        timeFrame = QtWidgets.QGroupBox()
        timeLayout = QtWidgets.QVBoxLayout(timeFrame)
        timeFrame.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))

        # Set up bandwidth label and spinbox
        bwLayout = QtWidgets.QHBoxLayout()
        bwLayout.addWidget(bandWidthLabel)
        bwLayout.addWidget(self.bandWidthSpinBox)
        bwLayout.addStretch()
        timeLayout.addLayout(bwLayout)

        # Set up datetime edits
        self.timeEdit = TimeEdit()
        self.timeEdit.setupMinMax(window.getMinAndMaxDateTime())
        timeLayout.addWidget(self.timeEdit.start)
        timeLayout.addWidget(self.timeEdit.end)
        bottomLayout.addWidget(timeFrame)

        # Set up options checkboxes
        optFrame = QtWidgets.QGroupBox()
        optLayout = QtWidgets.QGridLayout(optFrame)
        optLayout.addWidget(self.separateTracesCheckBox, 1, 0, 1, 2)
        optLayout.addWidget(self.aspectLockedCheckBox, 2, 0, 1, 2)
        optLayout.addWidget(self.logModeCheckBox, 3, 0, 1, 2)
        optLayout.addWidget(self.unitRatioCheckbox, 4, 0, 1, 2)
        bottomLayout.addWidget(optFrame)

        # setup dropdowns for coherence and phase pair selection
        wrapperLayout = QtWidgets.QVBoxLayout()
        cohPhaseFrame = QtWidgets.QGroupBox(' Coherence/Phase Pair')
        cohPhaseLayout = QtWidgets.QHBoxLayout(cohPhaseFrame)
        self.cohPair0 = QtWidgets.QComboBox()
        cohPhaseLayout.addWidget(self.cohPair0)
        self.cohPair1 = QtWidgets.QComboBox()
        cohPhaseLayout.addWidget(self.cohPair1)
        for dstrs in window.lastPlotStrings:
            for dstr,en in dstrs:
                self.cohPair0.addItem(dstr)
                self.cohPair1.addItem(dstr)
        if self.cohPair1.count() >= 2:
            self.cohPair1.setCurrentIndex(1)

        wrapperLayout.addWidget(cohPhaseFrame)
        self.combinedFrame = self.buildCombinedSpecFrame(window)
        wrapperLayout.addWidget(self.combinedFrame)

        bottomLayout.addLayout(wrapperLayout, 1)

        self.waveanalysisButton = QtWidgets.QPushButton('Open Wave Analysis')
        self.waveanalysisButton.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        self.updateButton = QtWidgets.QPushButton('Update')
        self.updateButton.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        # Create frame w/ vertical layout for update & wave buttons
        btnFrame = QtWidgets.QGroupBox()
        btnLayout = QtWidgets.QVBoxLayout(btnFrame)
        btnLayout.addWidget(self.waveanalysisButton)
        btnLayout.addWidget(self.updateButton)
        bottomLayout.addWidget(btnFrame)

        layout.addLayout(bottomLayout)
    
    def getVecGrps(self, dstrs):
        found = []
        for kw in ['BX','BY','BZ']:
            f = []
            for dstr in dstrs:
                if kw.lower() in dstr.lower():
                    f.append(dstr)
            found.append(f)
        return found

    def buildCombinedSpecFrame(self, window):
        # Set up UI elements
        frame = QtWidgets.QGroupBox(' Plot Sum of Powers')
        frame.setCheckable(True)
        frame.setChecked(False)
        frame.clicked.connect(self.greyOutSumOfPowers)
        self.greyOutSumOfPowers(False)
        layout = QtWidgets.QHBoxLayout(frame)
        ddBoxes = [QtWidgets.QComboBox() for i in range(0, 3)]
        self.axisBoxes = ddBoxes

        # Try to identify x-y-z groups
        dstrs = window.DATASTRINGS[:]
        found = self.getVecGrps(dstrs)

        # Check if this is a nonstandard file
        listLens = list(map(len, found))
        nonstandardFile = 0 in listLens

        # Add combo boxes to layout and fill accordingly
        for row in range(0, 3):
            box = ddBoxes[row]
            grp = found[row]
            if nonstandardFile:
                grp = dstrs
            
            for dstr in grp:
                box.addItem(dstr)
            
            if nonstandardFile:
                box.setCurrentIndex(row)

            layout.addWidget(box)

        # Try to set combo box indices to selected vector
        if not nonstandardFile:
            shownDstrs = [] # Get list of plotted dstrs
            for dstrLst in window.lastPlotStrings:
                for dstr, en in dstrLst:
                    shownDstrs.append(dstr)
            # Find groups in plotted dstrs
            shownGrps = self.getVecGrps(shownDstrs)
            listLens = list(map(len, shownGrps))
            # If at least one full vector is plotted through some combination
            # then adjust the index of the combo boxes
            if 0 not in listLens:
                firstSeen = shownGrps[0][0]
                index = found[0].index(firstSeen)
                for row in range(0, 3):
                    ddBoxes[row].setCurrentIndex(index)

        return frame

    def greyOutSumOfPowers(self, val):
        if val == False:
            self.tabs.setTabEnabled(3, False)
