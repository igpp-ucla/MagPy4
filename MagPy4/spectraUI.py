
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from .pyqtgraphExtensions import GridGraphicsLayout, SpectraGrid, BLabelItem
from .MagPy4UI import TimeEdit

class SpectraUI(object):
    def buildSpectraView(self):
        view = pg.GraphicsView()
        view.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        gmain = GridGraphicsLayout() # made this based off pg.GraphicsLayout
        gmain.setContentsMargins(11,0,11,0) # left top right bottom
        view.setCentralItem(gmain)
        grid = SpectraGrid(4)
        grid.setContentsMargins(0,0,0,0)
        labelLayout = GridGraphicsLayout()
        labelLayout.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Minimum))
        labelLayout.setContentsMargins(11, 3, 11, 5)
        gmain.addItem(grid, 0, 0, 1, 1)
        gmain.addItem(labelLayout, 1, 0, 1, 1)
        return view, grid, labelLayout

    def buildCombinedView(self):
        view = pg.GraphicsView()
        view.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        gmain = GridGraphicsLayout() # made this based off pg.GraphicsLayout
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

        # bandwidth label and spinbox
        bottomLayout = QtWidgets.QHBoxLayout()
        bandWidthLabel = QtGui.QLabel("Average Bandwidth:  ")
        self.bandWidthSpinBox = QtGui.QSpinBox()
        self.bandWidthSpinBox.setSingleStep(2)
        self.bandWidthSpinBox.setMinimum(1)
        self.bandWidthSpinBox.setProperty("value", 3)
        self.bandWidthSpinBox.setFixedWidth(50)

        # this will separate multiple traces on the same plot
        self.separateTracesCheckBox = QtGui.QCheckBox('Separate Traces')
        self.separateTracesCheckBox.setChecked(False)
        ###separateTraces = QtGui.QLabel("Separate Traces")
      
        self.aspectLockedCheckBox = QtGui.QCheckBox('Lock Aspect Ratio')
        self.aspectLockedCheckBox.setChecked(False)
        ###aspectLockedLabel = QtGui.QLabel("Lock Aspect Ratio")

        self.logModeCheckBox = QtGui.QCheckBox('Logarithmic scaling')
        self.logModeCheckBox.setChecked(True)

        self.unitRatioCheckbox = QtGui.QCheckBox('Unit Ratio')
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
        self.timeEdit = TimeEdit(QtGui.QFont("monospace", 10 if window.OS == 'windows' else 11))
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

        self.waveAnalysisButton = QtWidgets.QPushButton('Open Wave Analysis')
        self.waveAnalysisButton.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        self.updateButton = QtWidgets.QPushButton('Update')
        self.updateButton.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))

        # Create frame w/ vertical layout for update & wave buttons
        btnFrame = QtWidgets.QGroupBox()
        btnLayout = QtWidgets.QVBoxLayout(btnFrame)
        btnLayout.addWidget(self.waveAnalysisButton)
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

class SpectraViewBox(pg.ViewBox): # custom viewbox event handling
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)

    # overriding part of this function to get resizing to work correctly with
    # manual y range override and fixed aspect ratio settings
    def updateViewRange(self, forceX=False, forceY=False):
        tr = self.targetRect()
        bounds = self.rect()
        aspect = self.state['aspectLocked']
        if aspect is not False and 0 not in [aspect, tr.height(), bounds.height(), bounds.width()]:
            targetRatio = tr.width()/tr.height() if tr.height() != 0 else 1
            viewRatio = (bounds.width() / bounds.height() if bounds.height() != 0 else 1) / aspect
            viewRatio = 1 if viewRatio == 0 else viewRatio
            if viewRatio > targetRatio:
                pg.ViewBox.updateViewRange(self,False,True) 
                return
        pg.ViewBox.updateViewRange(self,forceX,forceY) #default