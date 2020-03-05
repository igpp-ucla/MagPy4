
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

        self.layout.insertWidget(0, self.gview)

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

        # Get height factors, filling in with 1's for any added or modified
        # plots
        hfactors = self.factors
        while len(hfactors) < len(self.plotItems):
            hfactors.append(1)

        # Scale the factors by the sum of the factors, so we get the fraction
        # of the plot area height that should be assigned to each plot item
        totFactors = sum(hfactors)
        hfactors = np.array(hfactors) / totFactors

        # Determine new height for each plot
        botmAxisHeight = self.plotItems[-1].getAxis('bottom').maximumHeight() + 1
        labelSetHeight = 0
        if self.labelSetGrd:
            labelSetHeight = self.labelSetGrd.boundingRect().height()
        gridHeight = self.boundingRect().height()
        vertSpacing = self.layout.verticalSpacing() * (self.numPlots - 1)
        plotAreaHeight = gridHeight - vertSpacing - botmAxisHeight - labelSetHeight
        rowHeights = hfactors * plotAreaHeight

        # Update label font sizes and limit the column width
        maxFontSize = self.getFontSize(plotAreaHeight, self.numPlots)
        for lbl, rowHeight in zip(self.labels, rowHeights):
            lbl.layout.setContentsMargins(10, 0, 0, 0)
            lbl.adjustLabelSizes(maxFontSize, rowHeight)

        maxLabelWidth = self.getMaxLabelWidth(maxFontSize)
        self.layout.setColumnMaximumWidth(0, maxLabelWidth)

        # Look for color plots and adjust legends + legend labels accordingly
        for i in range(0, len(self.plotItems)):
            cpKw = self.colorPltKws[i]
            if cpKw is None:
                continue

            # Extract corresp. objects and get the row height for this plot item
            rowHeight = rowHeights[i]
            colorPlotInfo = self.colorPltInfo[cpKw]
            gradLegend = colorPlotInfo['Legend']
            gradLbl = colorPlotInfo['LegendLbl']

            # Resize legend label text
            if gradLbl:
                gradLbl.adjustLabelSizes(rowHeight)
                gradLbl.setContentsMargins(0, 0, 0, 0)

            # Update legend offsets
            gradLegend.setOffsets(1, 1, 1)

        # Set row heights, add in extra space for last plot's dateTime axis
        index = 0
        for row in range(self.startRow, self.startRow+self.numPlots-1):
            self.layout.setRowPreferredHeight(row, rowHeights[index])
            index += 1
        self.layout.setRowPreferredHeight(self.startRow+self.numPlots-1, rowHeights[-1] + botmAxisHeight)

        # Adjust vertical placement of bottom label
        botmLabel = self.labels[-1]
        lm, tm, rm, bm = botmLabel.layout.getContentsMargins()
        botmLabel.layout.setContentsMargins(lm, tm, rm, botmAxisHeight)

        # Additional adjustments to margins for color plot elements
        # if the last plot is a color plot
        if self.colorPltKws[-1] is not None:
            colorPlotInfo = self.colorPltInfo[self.colorPltKws[-1]]
            botmLabel = colorPlotInfo['LegendLbl']
            botmGrad = colorPlotInfo['Legend']
            lm, rm, tm, bm = botmLabel.layout.getContentsMargins()
            botmLabel.layout.setContentsMargins(lm, tm, rm, botmAxisHeight)
            botmGrad.setOffsets(1, botmAxisHeight, 2)

        self.adjustPlotWidths()

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

        # Inserts a plot and its label item into the grid
        lblCol, pltCol = 0, 1
        self.addItem(lbl, self.startRow + index, lblCol, 1, 1)
        self.addItem(plt, self.startRow + index, pltCol, 1, 1)

        plt.getAxis('bottom').setStyle(showValues=True)
        for axis in ['left','bottom', 'top', 'right']:
            plt.getAxis(axis).setStyle(tickLength=-5)

        self.labels.insert(index, lbl)
        self.plotItems.insert(index, plt)
        self.colorPltKws.insert(index, None)
        self.numPlots += 1
        self.setTimeLabel()

    def addColorPlt(self, plt, lblStr, colorBar, colorLbl=None, units=None,
                    colorLblSpan=1, index=None):
        lbl = StackedLabel([lblStr], ['#000000'], units=units)
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
        
        self.resizeEvent(None)

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
            self.labelSetLabel = StackedLabel([],[])
            self.labelSetLabel.setContentsMargins(0, 0, 0, 2)
            self.labelSetLabel.layout.setVerticalSpacing(-2)
            self.addItem(self.labelSetGrd, 0, 1, 1, 1)
            self.addItem(self.labelSetLabel, 0, 0, 1, 1)

        # Add dstr to labelSet label and align it to the right
        self.labelSetLabel.addSubLabel(dstr, '#000000')
        self.labelSetLabel.subLabels[-1].setAttr('justify', 'right')
        self.labelSetLabel.subLabels[-1].setText(dstr)
        # Create an axis for the dstr label set
        self.labelSetGrd.addLabelSet(dstr)

        # Initialize tick label locations/strings
        ticks = self.plotItems[0].getAxis('bottom')._tickLevels
        if ticks == None:
            return
        self.labelSetGrd.updateTicks(ticks)
        self.resizeEvent(None) # Update widths and adjust for new heights

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
            maxWidth = max(maxWidth, la.calcDesiredWidth())

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
        self.resizeEvent(None)

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

class StackedLabel(pg.GraphicsLayout):
    def __init__(self, dstrs, colors, units=None, window=None, *args, **kwargs):
        self.subLabels = []
        self.dstrs = dstrs
        self.colors = colors
        self.units = units
        pg.GraphicsLayout.__init__(self, *args, **kwargs)

        # Spacing/margins setup
        self.layout.setVerticalSpacing(-4)
        self.layout.setContentsMargins(10, 0, 0, 0)
        self.layout.setRowStretchFactor(0, 1)
        self.layout.setColumnAlignment(0, QtCore.Qt.AlignHCenter)
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

    def adjustLabelSizes(self, maxFontSize, rowHeight):
        if self.subLabels == []:
            return

        # Get the number of rows in this stacked label
        numRows = len(self.dstrs)
        if self.units:
            numRows += 1

        # Decrement the font size until the (font height * numRows) is less
        # than the maximum row height
        font = QtGui.QFont()
        font.setPointSizeF(maxFontSize)
        fontInfo = QtGui.QFontMetricsF(font)
        while (fontInfo.height() * numRows + 4) > rowHeight:
            font.setPointSizeF(font.pointSizeF() - 0.5)
            fontInfo = QtGui.QFontMetricsF(font)
            if font.pointSize() <= 2:
                break
        fontSize = font.pointSize()

        # Determines new font size and updates all labels accordingly
        for lbl, txt, clr in zip(self.subLabels, self.dstrs, self.colors):
            lbl.setText(txt, color=clr, size=str(fontSize)+'pt')

# Checks if libraries required to use MMS Orbit tool are installed
def checkForOrbitLibs():
    installed = True
    pkgs = ['requests', 'cdflib']
    for pkg in pkgs:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            installed = False
    
    return installed
