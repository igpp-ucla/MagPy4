
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from .addTickLabels import LabelSetGrid
import importlib.util

import numpy as np
import pyqtgraph as pg
import functools

from .pyqtgraphExtensions import GridGraphicsLayout,BLabelItem
from . import getRelPath
from .mth import Mth

class MagPy4UI(object):

    def setupUI(self, window):
        self.window = window
        # gives default window options in top right
        window.setWindowFlags(QtCore.Qt.Window)
        window.resize(1280, 700)

        self.centralWidget = QtWidgets.QWidget(window)
        window.setCentralWidget(self.centralWidget)

        # Define actions.

        self.actionOpenFF = QtWidgets.QAction(window)
        self.actionOpenFF.setText('&Open Flat File...')
        self.actionOpenFF.setShortcut('Ctrl+O')
        self.actionOpenFF.setStatusTip('Opens a flat file')

        self.actionAddFF = QtWidgets.QAction(window)
        self.actionAddFF.setText('&Add Flat File...')
        self.actionAddFF.setStatusTip('Adds a flat file')

        self.actionOpenCDF = QtWidgets.QAction(window)
        self.actionOpenCDF.setText('Open &CDF File...')
        self.actionOpenCDF.setStatusTip('Opens a CDF file (currently experimental)')

        self.actionOpenASCII = QtWidgets.QAction(window)
        self.actionOpenASCII.setText('Open ASCII File...')
        self.actionOpenASCII.setStatusTip('Opens a simple ASCII file')

        self.actionExportFF = QtWidgets.QAction(window)
        self.actionExportFF.setText('Export Flat File...')
        self.actionExportFF.setStatusTip('Exports current flat file with edited data')

        self.actionExit = QtWidgets.QAction(window)
        self.actionExit.setText('E&xit')
        self.actionExit.setStatusTip('Closes the program\'s window and quits the program')
        self.actionOpenWs = QtWidgets.QAction('Open Workspace...')
        self.actionSaveWs = QtWidgets.QAction('Save Workspace As...')

        self.actionShowData = QtWidgets.QAction(window)
        self.actionShowData.setText('&Data...')
        self.actionShowData.setStatusTip('Shows the loaded data in a table view')

        self.actionPlotMenu = QtWidgets.QAction(window)
        self.actionPlotMenu.setText('&Plot Menu...')
        self.actionPlotMenu.setStatusTip('Opens the plot menu')

        self.actionSpectra = QtWidgets.QAction(window)
        self.actionSpectra.setText('&Spectra...')
        self.actionSpectra.setStatusTip('Opens spectral analysis window')

        self.actionDynamicSpectra = QtWidgets.QAction(window)
        self.actionDynamicSpectra.setText('Dynamic Spectrogram...')
        self.actionDynamicSpectra.setStatusTip('Generates a dynamic spectrogram based on user settings')

        self.actionDetrend = QtWidgets.QAction(window)
        self.actionDetrend.setText('Detrend...')
        self.actionDetrend.setStatusTip('Detrends selected data and allows user to perform additional analysis on it')

        self.actionDynamicCohPha = QtWidgets.QAction(window)
        self.actionDynamicCohPha.setText('Dynamic Coherence/Phase...')
        self.actionDynamicCohPha.setStatusTip('Dynamic analysis of coherence and phase between two variables')

        self.actionEdit = QtWidgets.QAction(window)
        self.actionEdit.setText('&Edit...')
        self.actionEdit.setStatusTip('Opens edit window that allows you to rotate the data with matrices')

        self.actionDynWave = QtWidgets.QAction(window)
        self.actionDynWave.setText('Dynamic Wave Analysis...')
        self.actionDynWave.setStatusTip('Dynamic analysis of various wave analysis results')

        self.actionTraj = QtWidgets.QAction(window)
        self.actionTraj.setText('Trajectory Analysis...')
        self.actionTraj.setStatusTip('Opens trajectory analysis window')

        # MMS Tools
        self.actionPlaneNormal = QtWidgets.QAction(window)
        self.actionPlaneNormal.setText('Plane Normal...')
        self.actionPlaneNormal.setStatusTip('Calculates the normal to the plane using the timing method')

        self.actionCurlometer = QtWidgets.QAction(window)
        self.actionCurlometer.setText('Curlometer...')
        self.actionCurlometer.setStatusTip('Estimates the electric current density inside the tetrahedron')

        self.actionCurvature = QtWidgets.QAction(window)
        self.actionCurvature.setText('Curvature...')
        self.actionCurvature.setStatusTip('Calculates the curvature of the magnetic field at the mesocenter')

        self.actionEPAD = QtWidgets.QAction(window)
        self.actionEPAD.setText('Plot Electron Pitch Angle...')
        self.actionEPAD.setStatusTip('Plots a color-mapped representation of the electron pitch-angle distribution')

        self.actionEOmni = QtWidgets.QAction(window)
        self.actionEOmni.setText('Plot Electron/Ion Spectrum...')
        self.actionEOmni.setStatusTip('Plots a color-mapped representation of omni-directional electron/ion energy spectrum')

        self.actionMMSPressure = QtWidgets.QAction(window)
        self.actionMMSPressure.setText('Pressure...')
        self.actionMMSPressure.setStatusTip('Calculates magnetic, thermal, and total pressure')

        self.actionMMSOrbit = QtWidgets.QAction(window)
        self.actionMMSOrbit.setText('MMS Orbit...')
        self.actionMMSOrbit.setStatusTip('Plots MMS spacecraft orbit')

        self.actionMMSFormation = QtWidgets.QAction(window)
        self.actionMMSFormation.setText('MMS Formation...')
        self.actionMMSFormation.setStatusTip('Plots spacecraft formation in 3D view or as a 2D projection')

        # Selection menu actions
        self.actionFixSelection = QtWidgets.QAction(window)
        self.actionFixSelection.setText('Fix Selection...')
        self.actionFixSelection.setStatusTip('Saves currently selected region to use with other tools')
        self.actionFixSelection.setVisible(False)

        self.actionSelectByTime = QtWidgets.QAction(window)
        self.actionSelectByTime.setText('Select By Time...')
        self.actionSelectByTime.setStatusTip('Select a time region to apply the currently selected tool to')
        self.actionSelectByTime.setVisible(True)

        self.actionSelectView = QtWidgets.QAction(window)
        self.actionSelectView.setText('Select Visible Region')
        self.actionSelectView.setStatusTip('Select currently visible region')
        self.actionSelectView.setVisible(True)
        ## Select view is given the 'Ctrl+a' shortcut, typically used for 'Select All'
        self.actionSelectView.setShortcut('Ctrl+a')

        self.actionBatchSelect = QtWidgets.QAction(window)
        self.actionBatchSelect.setText('Batch Select...')
        self.actionBatchSelect.setToolTip('Select multiple regions to examine and apply tools on')

        # Options menu actions
        self.scaleYToCurrentTimeAction = QtWidgets.QAction('&Scale Y-range to Current Time Selection',checkable=True,checked=True)
        self.scaleYToCurrentTimeAction.setStatusTip('')
        self.antialiasAction = QtWidgets.QAction('Smooth &Lines (Antialiasing)',checkable=True,checked=True)
        self.antialiasAction.setStatusTip('')
        self.bridgeDataGaps = QtWidgets.QAction('&Bridge Data Gaps', checkable=True, checked=False)
        self.bridgeDataGaps.setStatusTip('')
        self.drawPoints = QtWidgets.QAction('&Draw Points (Unoptimized)', checkable=True, checked=False)
        self.drawPoints.setStatusTip('')
        self.downsampleAction = QtWidgets.QAction('Downsample Plot Data', checkable=True)
        self.downsampleAction.setStatusTip('Reduces number of visible samples; improves performance for large datasets')

        self.showFileLbl = QtWidgets.QAction('Show Filename Label', checkable=True)
        self.showFileLbl.setChecked(True)
        self.enableScrollingAction = QtWidgets.QAction('Enable Scrolling', checkable=True)
        self.enableScrollingAction.setStatusTip('Enables vertical scrolling of plot window and sets a minimum height for it')

        self.actionHelp = QtWidgets.QAction(window)
        self.actionHelp.setText('MagPy4 &Help')
        self.actionHelp.setShortcut('F1')
        self.actionHelp.setStatusTip('Opens help window with information about the program modules')

        self.actionAbout = QtWidgets.QAction(window)
        self.actionAbout.setText('&About MagPy4')
        self.actionAbout.setStatusTip('Displays the program\'s version number and copyright notice')

        self.actionChkForUpdt = QtWidgets.QAction(window)
        self.actionChkForUpdt.setText('Update MagPy4')

        self.runTests = QtWidgets.QAction(window)
        self.runTests.setText('Run Tests')
        self.runTests.setStatusTip('Runs unit tests for code')

        self.switchMode = QtWidgets.QAction(window)
        self.switchMode.setText('Switch to MarsPy')
        self.switchMode.setToolTip('Loads various presets specific to the Insight mission')

        # Build the menu bar.
        self.menuBar = window.menuBar()

        self.fileMenu = self.menuBar.addMenu('&File')
        self.fileMenu.addAction(self.actionOpenFF)
        self.fileMenu.addAction(self.actionAddFF)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionOpenASCII)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.switchMode)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionOpenWs)
        self.fileMenu.addAction(self.actionSaveWs)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionExit)

        self.toolsMenu = self.menuBar.addMenu('&Tools')
        self.toolsMenu.addAction(self.actionShowData)
        self.toolsMenu.addAction(self.actionPlotMenu)
        self.toolsMenu.addAction(self.actionEdit)
        self.toolsMenu.addAction(self.actionSpectra)
        self.toolsMenu.addAction(self.actionDetrend)
        self.toolsMenu.addAction(self.actionTraj)
        self.toolsMenu.addSeparator()
        self.toolsMenu.addAction(self.actionDynamicSpectra)
        self.toolsMenu.addAction(self.actionDynamicCohPha)
        self.toolsMenu.addAction(self.actionDynWave)

        self.MMSMenu = self.menuBar.addMenu('&MMS Tools')
        self.MMSMenu.addAction(self.actionPlaneNormal)
        self.MMSMenu.addAction(self.actionCurlometer)
        self.MMSMenu.addAction(self.actionCurvature)
        self.MMSMenu.addAction(self.actionEPAD)
        self.MMSMenu.addAction(self.actionEOmni)
        self.MMSMenu.addAction(self.actionMMSPressure)
        self.MMSMenu.addSeparator()

        if checkForOrbitLibs():
            self.MMSMenu.addAction(self.actionMMSOrbit)
            self.MMSMenu.addAction(self.actionMMSFormation)

        self.selectMenu = self.menuBar.addMenu('Selection Tools')
        self.selectMenu.addAction(self.actionFixSelection)
        self.selectMenu.addAction(self.actionSelectByTime)
        self.selectMenu.addAction(self.actionSelectView)
        self.selectMenu.addAction(self.actionBatchSelect)

        self.optionsMenu = self.menuBar.addMenu('&Options')
        self.optionsMenu.addAction(self.scaleYToCurrentTimeAction)
        self.optionsMenu.addAction(self.antialiasAction)
        self.optionsMenu.addAction(self.bridgeDataGaps)
        self.optionsMenu.addAction(self.drawPoints)
        self.optionsMenu.addAction(self.downsampleAction)
        self.optionsMenu.addSeparator()
        self.optionsMenu.addAction(self.showFileLbl)
        self.optionsMenu.addAction(self.enableScrollingAction)

        self.helpMenu = self.menuBar.addMenu('&Help')
        self.helpMenu.addAction(self.actionHelp)
        self.helpMenu.addSeparator()
        self.helpMenu.addAction(self.actionAbout)
        self.helpMenu.addSeparator()
        self.helpMenu.addAction(self.actionChkForUpdt)

        self.plotApprAction = QtWidgets.QAction(window)
        self.plotApprAction.setText('Change Plot Appearance...')

        self.addTickLblsAction = QtWidgets.QAction(window)
        self.addTickLblsAction.setText('Extra Tick Labels...')

        layout = QtWidgets.QVBoxLayout(self.centralWidget)

        # SLIDER setup
        sliderLayout = QtWidgets.QGridLayout() # r, c, w, h
        self.startSlider = QtWidgets.QSlider()
        self.startSlider.setOrientation(QtCore.Qt.Horizontal)
        self.endSlider = QtWidgets.QSlider()
        self.endSlider.setOrientation(QtCore.Qt.Horizontal)

        self.timeEdit = TimeEdit(QtGui.QFont("monospace", 11))

        # Create buttons for moving plot windows L or R by a fixed amt
        self.mvLftBtn = QtWidgets.QPushButton('<', window)
        self.mvRgtBtn = QtWidgets.QPushButton('>', window)
        self.mvLftBtn.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.mvRgtBtn.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.mvLftBtn.setFixedWidth(50)
        self.mvRgtBtn.setFixedWidth(50)
        self.mvLftBtn.setFont(QtGui.QFont('monospace', 11))
        self.mvRgtBtn.setFont(QtGui.QFont('monospace', 11))

        # Setup shortcuts to shift win w/ L/R keys
        self.mvLftShrtct = QtWidgets.QShortcut('Left', window)
        self.mvRgtShrtct = QtWidgets.QShortcut('Right', window)

        # Zoom in/out and 'view all' shortcuts
        self.zoomInShrtct = QtWidgets.QShortcut('Ctrl+=', window)
        self.zoomOutShrtct = QtWidgets.QShortcut('Ctrl+-', window)
        self.zoomAllShrtct = QtWidgets.QShortcut('Ctrl+0', window)

        # Shortcuts used when scrolling is enabled
        self.scrollPlusShrtct = QtWidgets.QShortcut('Ctrl+Shift+=', window)
        self.scrollMinusShrtct = QtWidgets.QShortcut('Ctrl+Shift+-', window)

        # Shift percentage box setup
        self.shftPrcntBox = QtWidgets.QSpinBox()
        self.shftPrcntBox.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)        
        self.shftPrcntBox.setRange(5, 100)
        self.shftPrcntBox.setSingleStep(10)
        self.shftPrcntBox.setValue(25) # Default is 1/4th of time range
        self.shftPrcntBox.setSuffix('%')
        self.shftPrcntBox.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.shftPrcntBox.setFont(QtGui.QFont('monospace', 11))

        # Status bar setup
        self.statusBar = window.statusBar()

        sliderLayout.addWidget(self.timeEdit.start, 0, 0, 1, 1)
        sliderLayout.addWidget(self.startSlider, 0, 1, 1, 1)
        sliderLayout.addWidget(self.mvLftBtn, 0, 2, 2, 1)
        sliderLayout.addWidget(self.mvRgtBtn, 0, 3, 2, 1)
        sliderLayout.addWidget(self.shftPrcntBox, 0, 4, 2, 1)
        sliderLayout.addWidget(self.timeEdit.end, 1, 0, 1, 1)
        sliderLayout.addWidget(self.endSlider, 1, 1, 1, 1)

        layout.addLayout(sliderLayout)

        # Set up start screen
        self.layout = layout
        self.startUp(window)

    # update slider tick amount and timers and labels and stuff based on new file
    def setupSliders(self, tick, max, minmax):
        #dont want to trigger callbacks from first plot
        self.startSlider.blockSignals(True)
        self.endSlider.blockSignals(True)

        self.startSlider.setMinimum(0)
        self.startSlider.setMaximum(max)
        self.startSlider.setTickInterval(tick)
        self.startSlider.setSingleStep(tick)
        self.startSlider.setValue(0)
        self.endSlider.setMinimum(0)
        self.endSlider.setMaximum(max)
        self.endSlider.setTickInterval(tick)
        self.endSlider.setSingleStep(tick)
        self.endSlider.setValue(max)

        self.timeEdit.setupMinMax(minmax)

        self.startSlider.blockSignals(False)
        self.endSlider.blockSignals(False)

    def enableUIElems(self, enabled=True):
        # Enable/disable all elems for interacting w/ plots
        elems = [self.startSlider, self.endSlider, self.shftPrcntBox, self.mvLftBtn,
                self.mvRgtBtn, self.timeEdit.start, self.timeEdit.end,
                self.mvLftShrtct, self.mvRgtShrtct, self.switchMode, self.MMSMenu,
                self.actionSaveWs]
        for e in elems:
            e.setEnabled(enabled)

    def showMMSMenu(self, visible):
        self.MMSMenu.menuAction().setVisible(visible)

    def showSelectionMenu(self, visible):
        self.selectMenu.menuAction().setVisible(visible)

    def startUp(self, window):
        # Create frame and insert it into main layout
        self.startFrame = QtWidgets.QFrame()
        self.layout.insertWidget(0, self.startFrame)

        # Set frame background to white
        startLt = QtWidgets.QVBoxLayout(self.startFrame)
        styleSheet = ".QFrame { background-color:" + '#ffffff' + '}'
        self.startFrame.setStyleSheet(styleSheet)

        # Set up main title, its font size, and color
        nameLabel = QtWidgets.QLabel('MarsPy / MagPy4')
        font = QtGui.QFont()
        font.setPointSize(40)
        font.setBold(True)
        font.setWeight(75)
        nameLabel.setFont(font)
        styleSheet = "* { color:" + '#f44242' + " }"
        nameLabel.setStyleSheet(styleSheet)

        descLbl = QtWidgets.QLabel('Magnetic Field Analysis Program')
        font = QtGui.QFont()
        font.setPointSize(16)
        descLbl.setFont(font)
        ownerLbl = QtWidgets.QLabel('IGPP / UCLA')
        font.setPointSize(12)
        ownerLbl.setFont(font)

        # Mode selection UI elements
        modeLbl = QtWidgets.QLabel('Mode: ')
        self.modeComboBx = QtWidgets.QComboBox()
        self.modeComboBx.addItem('MarsPy')
        self.modeComboBx.addItem('MagPy')
        self.modeComboBx.currentTextChanged.connect(self.switchMainMode)

        # Checkbox to save mode for next time
        self.saveModeChkBx = QtWidgets.QCheckBox('Set As Default')
        self.saveModeChkBx.clicked.connect(self.saveState)

        # Layout for mode box + openFFBtn
        modeFrame = QtWidgets.QFrame()
        modeLt = QtWidgets.QGridLayout(modeFrame)
        modeLt.addWidget(modeLbl, 0, 0, 1, 1)
        modeLt.addWidget(self.modeComboBx, 0, 1, 1, 1)
        modeLt.addWidget(self.saveModeChkBx, 1, 1, 1, 1)

        # Initialize default mode based on state file
        stateFileName = 'state.txt'
        statePath = getRelPath(stateFileName)
        try: # If file exists, use setting
            fd = open(statePath, 'r')
            mode = fd.readline()
            nameLabel.setText(mode.strip('\n'))
            fd.close()
        except: # Otherwise MagPy is default
            mode = 'MagPy'

        if mode.strip('\n').strip(' ') == 'MagPy':
            self.modeComboBx.setCurrentIndex(1)
            self.window.insightMode = False
        else:
            self.modeComboBx.setCurrentIndex(0)
            self.window.insightMode = True
        self.modeComboBx.currentTextChanged.connect(nameLabel.setText)

        # Button to open a flat file
        openFFBtn = QtWidgets.QPushButton('Open Flat File')
        openFFBtn.clicked.connect(self.actionOpenFF.triggered)
        modeLt.addWidget(openFFBtn, 0, 2, 1, 1)

        # Create a layout that will be centered within startLt
        frameLt = QtWidgets.QGridLayout()
        startLt.addStretch()
        startLt.addLayout(frameLt)
        startLt.addStretch()

        # Add all elements into layout
        alignCenter = QtCore.Qt.AlignCenter
        spacer1 = QtWidgets.QSpacerItem(0, 10)
        spacer2 = QtWidgets.QSpacerItem(0, 10)

        frameLt.addWidget(nameLabel, 0, 0, 1, 1, alignCenter)
        frameLt.addItem(spacer1, 1, 0, 1, 1, alignCenter)
        frameLt.addWidget(descLbl, 2, 0, 1, 1, alignCenter)
        frameLt.addWidget(ownerLbl, 3, 0, 1, 1, alignCenter)
        frameLt.addItem(spacer2, 4, 0, 1, 1, alignCenter)
        frameLt.addWidget(modeFrame, 5, 0, 1, 1, alignCenter)

        self.startLt = startLt

        # Disable UI elements for interacting with plots
        self.enableUIElems(False)

    def saveState(self):
        # Save the current text in the mode combo box to the state file
        if self.saveModeChkBx.isChecked():
            mode = self.modeComboBx.currentText()
            stateFileName = 'state.txt'
            statePath = getRelPath(stateFileName)
            fd = open(statePath, 'w')
            fd.write(mode)
            fd.close()

    def switchMainMode(self):
        # Set insightMode in window so it knows which defaults/settings to load
        mode = self.modeComboBx.currentText()
        if mode == 'MarsPy':
            self.window.insightMode = True
        else:
            self.window.insightMode = False
        self.saveState()

    def setupView(self):
        # Remove start up layout and setup main plot grid
        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout(self.window)
        self.gview.setCentralItem(self.glw)

        self.layout.removeWidget(self.startFrame)
        self.startFrame.deleteLater()

        # Wrap gview inside a scroll frame and insert into main layout
        self.scrollFrame = QtWidgets.QScrollArea()
        self.scrollFrame.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        self.scrollFrame.setWidget(self.gview)
        self.scrollFrame.setWidgetResizable(True)
        self.scrollFrame.setFrameStyle(QtWidgets.QFrame.NoFrame)

        self.layout.insertWidget(0, self.scrollFrame)

        # Re-enable all UI elements for interacting w/ plots
        self.enableUIElems(True)

class TimeEdit():
    def __init__(self, font):
        self.start = QtWidgets.QDateTimeEdit()
        self.end = QtWidgets.QDateTimeEdit()

        for edit in [self.start, self.end]:
            edit.setFont(font)
            edit.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
            edit.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")

        self.start.editingFinished.connect(functools.partial(self.enforceMinMax, self.start))
        self.end.editingFinished.connect(functools.partial(self.enforceMinMax, self.end))

        self.linkedRegion = None

    def setupMinMax(self, minmax):
        min, max = minmax
        self.minDateTime = min
        self.maxDateTime = max
        self.setStartNoCallback(min)
        self.setEndNoCallback(max)

    def setWithNoCallback(dte, dt):
        dte.blockSignals(True)
        dte.setDateTime(dt)
        dte.blockSignals(False)

    def setStartNoCallback(self, dt):
        TimeEdit.setWithNoCallback(self.start, dt)

    def setEndNoCallback(self, dt):
        TimeEdit.setWithNoCallback(self.end, dt)

    def setLinkedRegion(self, region):
        ''' 
        Set the linked region that should be updated when this time edit's
        values are changed
        '''
        self.linkedRegion = region

    def getLinkedRegion(self):
        return self.linkedRegion

    def removeLinkToSelect(self, func):
        ''' 
        Disconnects this time edit from its linked GeneralSelect object
        '''
        self.linkedRegion = None
        self.start.dateTimeChanged.disconnect(func)
        self.end.dateTimeChanged.disconnect(func)

    def linkToSelect(self, func):
        ''' 
        Link this time edit to a GeneralSelect object that will respond
        to changes in this time edit's values and apply them to
        this time edit's linked region if it has one
        '''
        self.start.dateTimeChanged.connect(func)
        self.end.dateTimeChanged.connect(func)

    # done this way to avoid mid editing corrections
    def enforceMinMax(self, dte):
        min = self.minDateTime
        max = self.maxDateTime
        dt = dte.dateTime()
        dte.setDateTime(min if dt < min else max if dt > max else dt)

    def toString(self):
        #form = "yyyy MM dd hh mm ss zzz"
        form = "yyyy MMM dd hh:mm:ss.zzz"
        d0 = self.start.dateTime().toString(form)
        d1 = self.end.dateTime().toString(form)
        return d0,d1    

class MatrixWidget(QtWidgets.QWidget):
    def __init__(self, type='labels', prec=None, parent=None):
        #QtWidgets.QWidget.__init__(self, parent)
        super(MatrixWidget, self).__init__(parent)
        grid = QtWidgets.QGridLayout(self)
        self.prec = prec
        self.mat = [] # matrix of label or line widgets
        grid.setContentsMargins(0,0,0,0)
        for y in range(3):
            row = []
            for x in range(3):
                if type == 'labels':
                    w = QtGui.QLabel('0.0')
                    w.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                elif type == 'lines':
                    w = QtGui.QLineEdit()
                    w.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly) #i dont even know if this does anything
                    w.setText('0.0')
                else:
                    assert False, 'unknown type requested in MatrixWidget!'
                grid.addWidget(w, y, x, 1, 1)
                row.append(w)
            self.mat.append(row)

        #self.update()

    def setMatrix(self, m):
        for i in range(3):
            for j in range(3):
                if self.prec == None:
                    self.mat[i][j].setText(Mth.formatNumber(m[i][j]))
                else:
                    if abs(np.min(m)) < 1/1000 or abs(np.max(m)) < 1/1000000:
                        self.mat[i][j].setText(np.format_float_scientific(m[i][j], precision=self.prec))
                    else:
                        self.mat[i][j].setText(np.array2string(m[i][j], precision=self.prec))
                self.mat[i][j].repaint() # seems to fix max repaint problems

    # returns list of numbers
    def getMatrix(self):
        M = Mth.empty()
        for i in range(3):
            for j in range(3):
                s = self.mat[i][j].text()
                try:
                    f = float(s)
                except ValueError:
                    print(f'matrix has non-number at location {i},{j}')
                    f = 0.0
                M[i][j] = f
        return M

    def toString(self):
        return Mth.matToString(self.getMatrix())

class VectorWidget(QtWidgets.QWidget):
    def __init__(self, prec=None):
        super(VectorWidget, self).__init__(None)
        vecLt = QtWidgets.QGridLayout(self)
        vecLt.setContentsMargins(0,0,0,0)
        self.lbls = []
        self.prec = prec
        for i in range(0, 3):
            lbl = QtWidgets.QLabel()
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            vecLt.addWidget(lbl, i, 0, 1, 1)
            self.lbls.append(lbl)
        self.setVector([0,0,0])

    def setVector(self, vec):
        for i in range(0, 3):
            if self.prec == None:
                self.lbls[i].setText(str(vec[i]))
            else:
                if abs(np.min(vec)) < 1/1000 or abs(np.max(vec)) > (10 ** (self.prec + 1)):
                    self.lbls[i].setText(np.format_float_scientific(vec[i], precision=self.prec))
                else:
                    self.lbls[i].setText(str(np.round(vec[i], decimals=self.prec)))

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

# pyqt utils
class PyQtUtils:
    def clearLayout(layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()

    def moveToFront(window):
        if window:
            # this will remove minimized status 
            # and restore window with keeping maximized/normal state
            #window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
            window.raise_()
            # this will activate the window
            window.activateWindow()

class PlotGrid(pg.GraphicsLayout):
    def __init__(self, window=None, *args, **kwargs):
        self.window = window
        self.numPlots = 0
        self.plotItems = []
        self.labels = []

        # Additional lists for handling color plots
        self.colorPltKws = []
        self.colorPltInfo = {}

        # Elements used if additional tick labels are added
        self.labelSetGrd = None
        self.labelSetLabel = None
        self.startRow = 1
        self.factors = []

        pg.GraphicsLayout.__init__(self, *args, **kwargs)
        self.layout.setVerticalSpacing(2)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setColumnStretchFactor(0, 0)
        self.layout.setRowStretchFactor(0, 0)

    def count(self):
        # Returns number of plots
        return self.numPlots

    def setHeightFactors(self, heightFactors):
        self.factors = heightFactors

    def setTimeLabel(self):
        if self.plotItems == []:
            return

        for plt in self.plotItems[:-1]:
            self.hideTickValues(plt)

        bottomAxis = self.plotItems[-1].getAxis('bottom')
        bottomAxis.setStyle(showValues=True)
        bottomAxis.setLabel(bottomAxis.getDefaultLabel())
    
    def hideTickValues(self, plt):
        bottomAxis = plt.getAxis('bottom')
        bottomAxis.showLabel(False)
        bottomAxis.setStyle(showValues=False)

    def numColorPlts(self):
        return len(self.colorPltInfo.keys())

    def resizeEvent(self, event):
        if self.numPlots == 0:
            pg.GraphicsLayout.resizeEvent(self, event)
            return

        # Fill in factors w/ 1's to make sure # of factors = # of plots
        while len(self.factors) < self.count():
            self.factors.append(1)

        # Set vertical stretch factors for each row
        for row in range(self.startRow, self.startRow + self.count()):
            self.layout.setRowStretchFactor(row, self.factors[row-self.startRow])

        # Get the plot grid height / width
        newSize = event.newSize() if event is not None else self.boundingRect().size()
        width, height = newSize.width(), newSize.height()

        # Estimate height for each plot
        numPlots = len(self.plotItems)
        totalFactors = sum(self.factors)
        plotRatios = [self.factors[i] / totalFactors for i in range(0, numPlots)]
        plotHeights = [height * ratio for ratio in plotRatios]

        # Make labels approx 1/15th of plot width
        lblWidth = max(int(width)/15, 50)

        # Resize plot labels + grad legend labels
        self.resizeLabels(lblWidth, plotHeights)

        # Adjust axis widths
        self.adjustPlotWidths()


    def resizeLabels(self, lblWidth, plotHeights):
        # Find the smallest estimated font size among all the left plot labels
        fontSize = None
        for i in range(0, len(self.labels)):
            plotHeight = plotHeights[i]
            lblFontSize = self.labels[i].estimateFontSize(lblWidth, plotHeight)

            if fontSize is None:
                fontSize = lblWidth
            fontSize = min(lblFontSize, fontSize)

        # Set the font size for all left plot labels to the minimum found above
        for lbl in self.labels:
            lbl.setFontSize(fontSize)

        del plotHeight
        del lblFontSize
        del fontSize

        # Find the smallest estimated font size for all the grad legend labels
        fontSize = None
        for plotName in self.colorPltInfo:
            # Get grad legend label object and its width
            info = self.colorPltInfo[plotName]
            lbl = info['LegendLbl']
            lblWidth = lbl.boundingRect().width()

            # Get the estimated plot height
            index = self.getColorPlotIndex(plotName)
            plotHeight = plotHeights[index]

            # Get the estimated font size and compare
            lblFontSize = lbl.estimateFontSize(lblWidth, plotHeight)
            if fontSize is None:
                fontSize = lblFontSize
            else:
                fontSize = min(lblFontSize, fontSize)

        if self.colorPltInfo != {}:
            del lbl
            del lblFontSize

        # Set the font size for all grad legend labels
        for plotName in self.colorPltInfo:
            info = self.colorPltInfo[plotName]
            lbl = info['LegendLbl']
            for sublbl in lbl.sublabels:
                sublbl.setAttr('size', f'{fontSize}pt')
                sublbl.setText(sublbl.text)

    def getFontSize(self, height, numPlots):
        # Hard-coded method for determing label font sizes based on window size
        fontSize = height / numPlots * 0.0775
        fontSize = min(18, max(fontSize, 4))
        return fontSize

    def getMaxLabelWidth(self, maxFontSize):
        # Calculate the length of the variable names in each stacked plot label
        # and find the maximum of these lengths
        lenList = [list(map(len, lbl.dstrs)) for lbl in self.labels]
        lenList = np.concatenate(lenList)
        maxChar = max(lenList)

        # Calculate the number of pixels needed to represent maxChar characters
        # using a font with the given maximum font size
        font = QtGui.QFont()
        font.setPointSizeF(maxFontSize)
        met = QtGui.QFontMetricsF(font)
        labelWidth = met.averageCharWidth() * maxChar
        return labelWidth

    def addPlt(self, plt, lbl, index=None):
        index = self.numPlots if index is None else index

        # Each time a new plot is added
        if self.numPlots > 0:
            # Remove the old plot and spacer item below it
            old_plt = self.layout.itemAt(self.startRow + index - 1, 1)
            old_spacer = self.layout.itemAt(self.startRow + index, 0)
            self.removeItem(old_plt)
            self.removeItem(old_spacer)

            # Re-add the old plot with a row span of 1 this time
            # and reset the maximum row height
            self.addItem(old_plt, self.startRow + index - 1, 1, 1, 1)
            self.layout.setRowMaximumHeight(self.startRow+index, 1e28)

        # Inserts a plot and its label item into the grid
        lblCol, pltCol = 0, 1
        self.addItem(lbl, self.startRow + index, lblCol, 1, 1)
        self.addItem(plt, self.startRow + index, pltCol, 2, 1)

        # Make sure tick styles and settings are properly set
        plt.getAxis('bottom').setStyle(showValues=True)
        for axis in ['left','bottom', 'top', 'right']:
            plt.getAxis(axis).setStyle(tickLength=-5)

        # Update state information
        self.labels.insert(index, lbl)
        self.plotItems.insert(index, plt)
        self.colorPltKws.insert(index, None)
        self.numPlots += 1
        self.setTimeLabel()

        # Add in a spacer item just below the plot label and set the
        # row height to be equal to the height of the bottom axis item
        spacer = pg.LabelItem('')
        self.addItem(spacer, self.startRow + index + 1, lblCol, 1, 1)

        botmHt = plt.getAxis('bottom').maximumHeight()
        self.layout.setRowMaximumHeight(self.startRow+index+1, botmHt)
        self.layout.setRowMinimumHeight(self.startRow+index+1, botmHt)

        self.adjustPlotWidths()

    def addColorPlt(self, plt, lblStr, colorBar, colorLbl=None, units=None,
                    colorLblSpan=1, index=None):
        lbl = StackedLabel([lblStr], ['#000000'], units=units)
        colorBar.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        colorBar.setMaximumHeight(1e28)
        colorBar.setOffsets(1, 1, 5, 0)
        self.addPlt(plt, lbl, index)
        index = self.numPlots - 1 if index is None else index
        self.addItem(colorBar, self.startRow + index, 2, 1, 1)
        if colorLbl:
            self.addItem(colorLbl, self.startRow + index, 3, 1, 1)

        # Additional state updates
        plt.getAxis('bottom').tickOffset = self.window.tickOffset
        colorBar.setBarWidth(28)

        # Add tracker lines to plots
        trackerLine = pg.InfiniteLine(movable=False, angle=90, pos=0, pen=pg.mkPen('#000000', width=1, style=QtCore.Qt.DashLine))
        plt.addItem(trackerLine)
        self.window.trackerLines.insert(index, trackerLine)

        # Update state information
        cpInfo = {}
        cpInfo['Legend'] = colorBar
        cpInfo['LegendLbl'] = colorLbl
        cpInfo['Units'] = units
        self.colorPltInfo[lblStr] = cpInfo
        self.colorPltKws[index] = lblStr

    def getColorPlotIndex(self, name):
        # Get row index corresponding to the color plot w/ given name,
        # returning None if it isn't in the grid
        index = None
        if name in self.colorPltInfo.keys():
            index = self.colorPltKws.index(name)

        return index

    def replaceColorPlot(self, pltIndex, newPlt, lblStr, colorBar, colorLbl=None, 
                        units=None):
        if pltIndex >= self.numPlots or self.colorPltKws[pltIndex] is None:
            return

        # Remove the grid items and objects from state info
        # and add the color plot to the grid
        self._removePlotFromGrid(pltIndex)
        self.addColorPlt(newPlt, lblStr, colorBar, colorLbl, units, index=pltIndex)

        # Adjust displayed tick marks and time axis labels
        self.setTimeLabel()

    def _removePlotFromGrid(self, pltIndex):
        if pltIndex >= self.numPlots or pltIndex < 0:
            return

        # Get plot items/labels and remove from lists
        lbl = self.labels.pop(pltIndex)
        plt = self.plotItems.pop(pltIndex)
        trackline = self.window.trackerLines.pop(pltIndex)
        trackline.deleteLater()
        self.numPlots -= 1

        # Additional handling if color plot
        if self.colorPltKws[pltIndex] is not None:
            kw = self.colorPltKws[pltIndex]
            colorPltInfo = self.colorPltInfo[kw]
            elems = [colorPltInfo['Legend'], colorPltInfo['LegendLbl']]
            for elem in elems:
                if elem is None:
                    continue
                self.removeItem(elem)
                elem.deleteLater()

        # Remove plot & label elements from layout
        for item in [lbl, plt]:
            self.removeItem(item)
            item.deleteLater()

        # Update color plot status
        self.colorPltKws.pop(pltIndex)

    def removePlot(self, pltIndex=None):
        if self.numPlots < 1:
            return

        if pltIndex is None:
            pltIndex = self.numPlots - 1

        # Remove plot and related objects from grid
        self._removePlotFromGrid(pltIndex)

        self.setTimeLabel()

    def addLabelSet(self, dstr):
        # Initialize label set grid/stacked label if one hasn't been created yet
        if self.labelSetGrd == None:
            self.labelSetGrd = LabelSetGrid(self.window)
            self.labelSetLabel = LabelSetLabel()
            self.labelSetLabel.layout.setContentsMargins(0, 0, 0, 0)
            self.labelSetLabel.layout.setVerticalSpacing(1)
            self.addItem(self.labelSetGrd, 0, 1, 1, 1)
            self.addItem(self.labelSetLabel, 0, 0, 1, 1)

        # Add dstr to labelSet label and align it to 
        self.labelSetLabel.addSubLabel(dstr, '#000000')
        for lbl in self.labelSetLabel.subLabels:
            lbl.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))

        # Create an axis for the dstr label set
        self.labelSetGrd.addLabelSet(dstr)

        # Adjust sublabel heights so they are aligned w/ the tick labels
        for lbl in self.labelSetLabel.subLabels:
            ht = 22
            lbl.setMinimumHeight(ht)
            lbl.setMaximumHeight(ht)

        # Initialize tick label locations/strings
        ticks = self.plotItems[0].getAxis('bottom')._tickLevels
        if ticks == None:
            return
        self.labelSetGrd.updateTicks(ticks)

    def removeLabelSet(self, dstr):
        if self.labelSetGrd == None:
            return
        else:
            # Get new dstrs list after removing, remove old items, and reset vals
            dstrs = self.labelSetLabel.dstrs[:]
            dstrs.remove(dstr)
            self.removeItem(self.labelSetLabel)
            self.removeItem(self.labelSetGrd)
            self.labelSetGrd = None
            self.labelSetLabel = None

            # Recreate label and label set grid if new dstrs list is not empty
            if dstrs != []:
                for dstr in dstrs:
                    self.addLabelSet(dstr)

    def adjustPlotWidths(self):
        # Get minimum width of all left axes
        maxWidth = 0
        for pi in self.plotItems:
            la = pi.getAxis('left')
            width = la.calcDesiredWidth()
            maxWidth = max(maxWidth, width)

        # Update all left axes widths
        for pi in self.plotItems:
            la = pi.getAxis('left')
            la.setWidth(maxWidth)

        if self.labelSetGrd: # Match extra tick axis widths to maxWidth
            self.labelSetGrd.adjustWidths(maxWidth)

    def adjustTitleColors(self, penList):
        # Get each pen list and stacked list corresponding to a plot
        for lbl, pltPenList in zip(self.labels, penList):
            subLblNum = 0
            # Match pen colors to sublabels in stacked label
            for dstrLabel, pen in zip(lbl.subLabels, pltPenList):
                color = pen.color()
                fontSize = dstrLabel.opts['size']
                dstrLabel.setText(dstrLabel.text, size=fontSize, color=color)
                # Update stacked label's list of colors to use when resizing
                lbl.colors[subLblNum] = color
                subLblNum += 1

    def setPlotLabel(self, lbl, plotNum):
        prevLabel = self.getPlotLabel(plotNum)
        self.removeItem(prevLabel)
        self.addItem(lbl, self.startRow+plotNum, 0, 1, 1)
        self.labels[plotNum] = lbl

    def getPlotLabel(self, plotNum):
        return self.labels[plotNum]

class MainPlotGrid(PlotGrid):
    def __init__(self, window=None, *args, **kwargs):
        PlotGrid.__init__(self, window, *args, **kwargs)
        self.menu = QtGui.QMenu()
        self.menu.addAction(self.window.ui.plotApprAction) # Plot appearance
        self.menu.addAction(self.window.ui.addTickLblsAction) # Additional labels

    def menuEnabled(self):
        return True

    def getContextMenus(self, event):
        return self.menu.actions() if self.menuEnabled() else []

class AdjustedLabel(pg.LabelItem):
    def getFont(self):
        return self.item.font()

    def getFontSize(self):
        if 'size' in self.opts:
            size = self.opts['size']
            return float(size[:-2])
        else:
            return self.getFont().pointSizeF()

    def setFontSize(self, pt):
        self.setAttr('size', pt)
        self.updateText()

    def setColor(self, color):
        self.setAttr('color', color)
        self.updateText()

    def updateText(self):
        self.setText(self.text)

    def updateMin(self):
        # Uses minimum font size to estimate minimum bounding
        # rect so resizeEvents work correctly
        bounds = self.itemRect()
        minWidth, minHeight = self.getMinimumSize()
        self.setMinimumWidth(minWidth)
        self.setMinimumHeight(minHeight)

        self._sizeHint = {
            QtCore.Qt.MinimumSize: (minWidth, minHeight),
            QtCore.Qt.PreferredSize: (bounds.width(), bounds.height()),
            QtCore.Qt.MaximumSize: (-1, -1),  #bounds.width()*2, bounds.height()*2),
            QtCore.Qt.MinimumDescent: (0, 0)  ##?? what is this?
        }
        self.updateGeometry()

    def getMinimumSize(self):
        # Estimates the minimum bounding rect for text w/
        # a font point size of 2
        min_font_size = 2
        font = self.getFont()
        font.setPointSize(min_font_size)
        fontMetrics = QtGui.QFontMetrics(font)
        minRect = fontMetrics.tightBoundingRect(self.text)

        return minRect.width(), minRect.height()

class StackedLabel(pg.GraphicsLayout):
    def __init__(self, dstrs, colors, units=None, window=None, *args, **kwargs):
        self.subLabels = []
        self.dstrs = dstrs
        self.colors = colors
        self.units = units
        pg.GraphicsLayout.__init__(self, *args, **kwargs)

        # Spacing/margins setup
        self.layout.setVerticalSpacing(-4)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setRowStretchFactor(0, 1)
        self.layout.setColumnAlignment(0, QtCore.Qt.AlignCenter)
        rowNum = 1

        # Add in every dstr label and set its color
        for dstr, clr in zip(dstrs, colors):
            curLabel = self.addLabel(text=dstr, color=clr, row=rowNum, col=0)
            curLabel.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
            self.subLabels.append(curLabel)
            rowNum += 1

        # Add unit label to bottom of stack
        if units is not None:
            unitStr = '[' + units + ']'
            unitLbl = self.addLabel(text=unitStr, color='#888888', row=rowNum, col=0)
            unitLbl.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
            self.subLabels.append(unitLbl)
            self.dstrs.append(unitStr)
            self.colors.append('#888888')
            rowNum += 1

        self.layout.setRowStretchFactor(rowNum, 1)

    def addLabel(self, text=' ', row=None, col=None, rowspan=1, colspan=1, **kargs):
        text = AdjustedLabel(text, justify='center', **kargs)
        self.addItem(text, row, col, rowspan, colspan)
        return text

    def addSubLabel(self, dstr, color):
        # Add another sublabel at end of stackedLabel
        lbl = self.addLabel(text=dstr, color=color, row=len(self.subLabels), col=0)

        # Reset stretch factor used to center the sublabels within the stackedLabel
        self.layout.setRowStretchFactor(len(self.subLabels), 0)
        self.layout.setRowStretchFactor(len(self.subLabels)+1, 1)

        # Update all internal lists
        self.subLabels.append(lbl)
        self.dstrs.append(dstr)
        self.colors.append(color)

    def setFontSize(self, pt):
        for label in self.subLabels:
            label.setText(label.text, size=f'{pt}pt')

    def count(self):
        return len(self.subLabels)

    def estimateFontSize(self, width, height):
        if len(self.subLabels) == 0:
            return QtGui.QFont().pointSize()

        # Set up font + font metrics information
        font = self.subLabels[0].getFont()
        fontSize = self.subLabels[0].getFontSize()
        font.setPointSize(fontSize)
        fontMetrics = QtGui.QFontMetricsF(font)

        # Estimate the current height/width of the text
        labelRects = [fontMetrics.boundingRect(lbl) for lbl in self.dstrs]
        estHeight = sum([fontMetrics.height()+2 for rect in labelRects])

        # Estimate the font size using the ratio of the rect width/height and
        # the estimated width/height 
        ratio = estHeight / (height*.875)
        newFontSize = int(min(max(2, fontSize / ratio), 16))

        # Adjust if not a multiple of 2
        if newFontSize % 2 != 0:
            newFontSize -= 1

        return newFontSize

class StackedAxisLabel(pg.GraphicsLayout):
    def __init__(self, lbls, angle=90, *args, **kwargs):
        self.lblTxt = lbls
        self.sublabels = []
        self.angle = angle
        pg.GraphicsLayout.__init__(self, *args, **kwargs)
        self.layout.setVerticalSpacing(-2)
        self.layout.setHorizontalSpacing(-2)
        if angle != 0:
            self.layout.setContentsMargins(0, 0, 0, 0)
        else:
            self.layout.setContentsMargins(0, 0, 0, 0)
        self.setupLabels(lbls)

    def getLabelText(self):
        return self.lblTxt

    def setupLabels(self, lbls):
        if self.angle > 0:
            lbls = lbls[::-1]
        if self.angle == 0 or self.angle == -180:
            self.layout.setRowStretchFactor(0, 1)
        for i in range(0, len(lbls)):
            lbl = lbls[i]
            sublbl = AdjustedLabel(lbl, angle=self.angle)
            if self.angle == 0 or self.angle == -180:
                self.addItem(sublbl, i+1, 0, 1, 1)
            else:
                self.addItem(sublbl, 0, i, 1, 1)
            self.sublabels.append(sublbl)
        if self.angle == 0 or self.angle == -180:
            self.layout.setRowStretchFactor(len(lbls)+1, 1)

    def getFont(self):
        # Returns the font w/ the correct point size set
        if self.sublabels == []:
            return QtGui.QFont()

        font = self.sublabels[0].item.font()
        if 'size' in self.sublabels[0].opts:
            fontSize = self.sublabels[0].opts['size'][:-2]
            fontSize = float(fontSize)
        else:
            fontSize = QtGui.QFont().pointSize()

        fontSize = min(fontSize, 12)
        font.setPointSize(fontSize)
        return font

    def estimateFontSize(self, width, height):
        # Set up font + font metrics information
        font = self.getFont()
        fontSize = font.pointSize()
        fontMetrics = QtGui.QFontMetricsF(font)

        # Estimate the current height/width of the text
        labelRects = [fontMetrics.boundingRect(lbl) for lbl in self.lblTxt]
        estHeight = max([rect.width() for rect in labelRects])

        # Estimate the font size using the ratio of the rect width/height and
        # the estimated width/height 
        ratio = estHeight / (height*.9)
        newFontSize = int(min(max(2, fontSize / ratio), 14))

        return newFontSize

class LabelSetLabel(pg.GraphicsLayout):
    def __init__(self, *args, **kwargs):
        self.dstrs = []
        self.subLabels = []
        self.colors = []

        pg.GraphicsLayout.__init__(self, *args, **kwargs)

    def addSubLabel(self, dstr, color):
        # Add another sublabel at end of stackedLabel
        lbl = self.addLabel(text=dstr, color=color, row=len(self.subLabels), col=0)

        # Update all internal lists
        self.subLabels.append(lbl)
        self.dstrs.append(dstr)
        self.colors.append(color)

# Checks if libraries required to use MMS Orbit tool are installed
def checkForOrbitLibs():
    installed = True
    pkgs = ['requests', 'cdflib']
    for pkg in pkgs:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            installed = False
    
    return installed
