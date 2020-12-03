
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from .addTickLabels import LabelSetGrid
import importlib.util

import numpy as np
import pyqtgraph as pg
import functools
import os

from .pyqtgraphExtensions import GridGraphicsLayout, BLabelItem
from .plotBase import GraphicsLayout, GraphicsView
from . import getRelPath
from .mth import Mth
import re
from . import config

class MagPy4UI(object):

    def setupUI(self, window):
        self.window = window
        # gives default window options in top right
        window.setWindowFlags(QtCore.Qt.Window)
        window.resize(1280, 700)

        frame = QtWidgets.QFrame(window)
        window.setCentralWidget(frame)

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

        self.actionExportDataFile = QtWidgets.QAction(window)
        self.actionExportDataFile.setText('Export Data File...')
        self.actionExportDataFile.setStatusTip('Exports current data in original file format (only data shape is same as original)')
        self.actionExportDataFile.setVisible(False)

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

        self.actionTraceStats = QtWidgets.QAction(window)
        self.actionTraceStats.setText('Trace Statistics')
        self.actionTraceStats.setStatusTip('Displays statistics about plotted variables')

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

        # Set up dynamic wave analysis actions
        keys = ['Azimuth Angle', 'Ellipticity (Means)', 'Ellipticity (SVD)',
            'Ellipticity (Born-Wolf)', 'Propagation Angle (Means)',
            'Propagation Angle (SVD)', 'Propagation Angle (Min Var)',
            'Power Spectra Trace', 'Compressional Power']
        keys = sorted(keys)
        self.dynWaveActions = []
        for key in keys:
            action = QtWidgets.QAction(window)
            action.setText(key + '...')
            self.dynWaveActions.append(action)

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

        self.actionFEEPSPAD = QtWidgets.QAction(window)
        self.actionFEEPSPAD.setText('Plot FEEPS Pitch Angle Distributions...')
        self.actionFEEPSPAD.setStatusTip('Plots the MMS FEEPS pitch angle distributions')

        self.actionMMSPressure = QtWidgets.QAction(window)
        self.actionMMSPressure.setText('Pressure...')
        self.actionMMSPressure.setStatusTip('Calculates magnetic, thermal, and total pressure')

        self.actionLoadMMS = QtWidgets.QAction(window)
        self.actionLoadMMS.setText('Load MMS Data...')
        self.actionLoadMMS.setStatusTip('Download MMS Data from specified intervals and data types')

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
        self.fileMenu.addAction(self.actionOpenCDF)
        self.fileMenu.addAction(self.actionOpenASCII)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionOpenWs)
        self.fileMenu.addAction(self.actionSaveWs)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionExportDataFile)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.switchMode)
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

        self.waveMenu = QtWidgets.QMenu('Dynamic Wave Analysis')
        for act in self.dynWaveActions:
            self.waveMenu.addAction(act)
        self.toolsMenu.addMenu(self.waveMenu)

        self.MMSMenu = self.menuBar.addMenu('&MMS Tools')
        self.MMSMenu.addAction(self.actionLoadMMS)
        self.MMSMenu.addAction(self.actionPlaneNormal)
        self.MMSMenu.addAction(self.actionCurlometer)
        self.MMSMenu.addAction(self.actionCurvature)
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

        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)

        # SLIDER setup
        sliderLayout = QtWidgets.QGridLayout() # r, c, w, h
        self.timeEdit = TimeEdit()

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

        # Enable/disable tracker
        self.toggleTrackerAction = QtWidgets.QAction(window)
        self.toggleTrackerAction.setText('Enable Tracker Line')
        self.toggleTrackerAction.setStatusTip('Shows line that follows mouse in plot window & updates timestamp in status bar')
        self.toggleTrackerAction.setCheckable(True)
        self.optionsMenu.addAction(self.toggleTrackerAction)

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
        margins = self.statusBar.layout().contentsMargins()
        margins.setBottom(5)
        self.statusBar.layout().setContentsMargins(margins)

        self.timeStatus = QtWidgets.QLabel()
        self.timeStatus.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        # self.timeStatus.setMinimumWidth(220)
        self.statusBar.addPermanentWidget(self.timeStatus)

        sliderLayout.addWidget(self.timeEdit.start, 0, 0, 1, 1)
        self.scrollSelect = ScrollSelector()
        sliderLayout.addWidget(self.scrollSelect, 0, 1, 2, 1)
        sliderLayout.addWidget(self.mvLftBtn, 0, 2, 2, 1)
        sliderLayout.addWidget(self.mvRgtBtn, 0, 3, 2, 1)
        sliderLayout.addWidget(self.shftPrcntBox, 0, 4, 2, 1)
        sliderLayout.addWidget(self.timeEdit.end, 1, 0, 1, 1)
        # sliderLayout.addWidget(self.endSlider, 1, 1, 1, 1)
        sliderLayout.setContentsMargins(10, 0, 10, 0)

        layout.addLayout(sliderLayout)

        # Set up start screen
        self.layout = layout
        self.startUp(window)

    # update slider tick amount and timers and labels and stuff based on new file
    def setupSliders(self, tick, max, minmax):
        self.scrollSelect.set_range(max)
        self.scrollSelect.set_start(0)
        self.scrollSelect.set_end(max)
        self.timeEdit.setupMinMax(minmax)

    def enableUIElems(self, enabled=True):
        # Enable/disable all elems for interacting w/ plots
        elems = [self.shftPrcntBox, self.mvLftBtn,
                self.mvRgtBtn, self.timeEdit.start, self.timeEdit.end,
                self.mvLftShrtct, self.mvRgtShrtct, self.switchMode, self.MMSMenu,
                self.actionSaveWs, self.scrollSelect]
        for e in elems:
            e.setEnabled(enabled)

    def showMMSMenu(self, visible):
        self.MMSMenu.menuAction().setVisible(visible)

    def showSelectionMenu(self, visible):
        self.selectMenu.menuAction().setVisible(visible)

    def getTitleFrame(self):
        ''' Set up frame that displays title of program
            and other information
        '''
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(10, 25, 10, 25)

        # Label text, font size, and colors
        labels = ['MarsPy / MagPy', 'Magnetic Field Analysis Program',
            'IGPP / UCLA']
        pt_sizes = [40, 16, 12]
        colors = [(15, 28, 61, 200), (50, 50, 50, 250), (60, 60, 60, 250)]

        # Create each label item and add to layout
        for lbl, pt, color in zip(labels, pt_sizes, colors):
            label = QtWidgets.QLabel(lbl)

            # Font size
            font = QtGui.QFont()
            font.setPointSize(pt)
            label.setFont(font)

            # Color
            style = f' * {{ color: rgba{str(color)}; }}'
            label.setStyleSheet(style)

            layout.addWidget(label, alignment=QtCore.Qt.AlignHCenter)

        return frame

    def getModeFrame(self):
        # Mode selection UI elements
        modeLbl = QtWidgets.QLabel('Mode: ')
        self.modeComboBx = QtWidgets.QComboBox()
        self.modeComboBx.addItem('MarsPy')
        self.modeComboBx.addItem('MagPy')
        self.modeComboBx.currentTextChanged.connect(self.switchMainMode)
        self.modeComboBx.setMinimumHeight(28)
        self.modeComboBx.setMinimumWidth(100)

        # Checkbox to save mode for next time
        self.saveModeChkBx = QtWidgets.QCheckBox('Set As Default')
        self.saveModeChkBx.clicked.connect(self.saveState)

        # Layout for mode box + openFFBtn
        modeFrame = QtWidgets.QFrame()
        modeLt = QtWidgets.QHBoxLayout(modeFrame)
        modeLt.addStretch()
        modeLt.addWidget(modeLbl)
        modeLt.addWidget(self.modeComboBx)
        modeLt.addWidget(self.saveModeChkBx)
        modeLt.addStretch()

        # Initialize default mode based on state file
        state_dict = self.window.readStateFile()
        if 'mode' in state_dict:
            mode = state_dict['mode']
        else:
            mode = 'MagPy'

        if mode == 'MagPy':
            self.modeComboBx.setCurrentIndex(1)
            self.window.insightMode = False
        else:
            self.modeComboBx.setCurrentIndex(0)
            self.window.insightMode = True

        return modeFrame

    def getButtonFrame(self):
        ''' Returns frame for buttons to open the various file types'''
        btnFrm = QtWidgets.QFrame()
        btnLt = QtWidgets.QHBoxLayout(btnFrm)

        # Button labels and corresponding actions
        labels = ['Open Flat File', 'Open ASCII File', 'Open CDF', 'Load MMS Data']
        actions = [self.actionOpenFF, self.actionOpenASCII, self.actionOpenCDF,
                self.actionLoadMMS]
        colors = ['#4a5582', '#804d4d', '#9e8d54', '#54999e']

        # Create each button, add to layout, and connect to action
        for i in range(0, len(labels)):
            # Button object set up
            btn = QtWidgets.QPushButton(' ' + labels[i] + ' ')
            btn.setMinimumHeight(30)
            btnLt.addWidget(btn)

            # Connect to action
            btn.clicked.connect(actions[i].triggered)

            # Set color
            style = f'''QPushButton {{
                background-color: {colors[i]};
                color: white;
                }}
            '''
            btn.setStyleSheet(style)

        return btnFrm

    def startUp(self, window):
        # Create frame and insert it into main layout
        self.startFrame = QtWidgets.QFrame()
        self.startFrame.setObjectName('background_frame')
        self.layout.insertWidget(0, self.startFrame)
        startLt = QtWidgets.QVBoxLayout(self.startFrame)

        # Set background image
        img_path = getRelPath('magpy_background.png')
        if os.name == 'nt':
            img_path = img_path.replace('\\', '/')

        style = f'''
            #background_frame {{
                border-image: url({img_path}) 0 0 0 0 stretch stretch;
            }}
        '''
        self.startFrame.setStyleSheet(style)

        # Create a layout that will be centered within startLt
        centerFrame = QtWidgets.QFrame()
        style = '''.QFrame { 
            background-color: rgba(255, 255, 255, 125);
            border-radius: 2px;
        }
        '''
        centerFrame.setStyleSheet(style)
        centerFrame.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))

        # Add main widgets to center layout
        centerLt = QtWidgets.QVBoxLayout(centerFrame)
        centerLt.setContentsMargins(25, 25, 25, 25)
        centerLt.setSpacing(14)

        titleFrm = self.getTitleFrame()
        modeFrm = self.getModeFrame()
        btnFrm = self.getButtonFrame()
        for elem in [titleFrm, modeFrm, btnFrm]:
            centerLt.addWidget(elem)

        # Center horizontally
        hwrapLt = QtWidgets.QHBoxLayout()
        hwrapLt.addStretch()
        hwrapLt.addWidget(centerFrame)
        hwrapLt.addStretch()

        # Add wrapper layout and center vertically
        startLt.addStretch()
        startLt.addLayout(hwrapLt)
        startLt.addStretch()

        self.startLt = startLt

        # Disable UI elements for interacting with plots
        self.enableUIElems(False)

    def saveState(self):
        # Save the current text in the mode combo box to the state file
        if self.saveModeChkBx.isChecked():
            mode = self.modeComboBx.currentText()
            self.window.updateStateFile('mode', mode)

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
        self.gview = GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout(self.window)
        self.glw.layout.setVerticalSpacing(0)
        self.glw.layout.setContentsMargins(5, 5, 5, 5)
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
    def __init__(self, font=None):
        if font is None:
            font_name, font_size = config.fonts['monospace']
            font = QtGui.QFont(font_name, font_size)

        self.start = QtWidgets.QDateTimeEdit()
        self.end = QtWidgets.QDateTimeEdit()

        for edit in [self.start, self.end]:
            edit.setFont(font)
            edit.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
            edit.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")

        self.start.editingFinished.connect(functools.partial(self.enforceMinMax, self.start))
        self.end.editingFinished.connect(functools.partial(self.enforceMinMax, self.end))

        self.linkedRegion = None

    def setFont(self, font):
        for edit in [self.start, self.end]:
            edit.setFont(font)

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
    
    def setTimeRange(self, start_dt, end_dt):
        self.start.setDateTime(start_dt)
        self.end.setDateTime(end_dt)
    
    def getRange(self):
        ''' Return start/end datetimes '''
        start = self.start.dateTime().toPyDateTime()
        end = self.end.dateTime().toPyDateTime()
        return (start, end)

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

class ScientificSpinBox(QtWidgets.QDoubleSpinBox):
    def validate(self, txt, pos):
        # Checks if string matches scientific notation format or is a regular num
        state = 1
        if re.fullmatch('\d*\.*\d*', txt):
            state = 2
        elif re.fullmatch('\d*.*\d*e\+\d+', txt) is not None:
            state = 2
        elif re.match('\d+.*\d*e', txt) or re.match('\d+.*\d*e\+', txt):
            state = 1
        else:
            state = 0

        # Case where prefix is set to '10^'
        if self.prefix() == '10^':
            if re.match('10\^\d*\.*\d*', txt) or re.match('10\^-\d*\.*\d*', txt):
                state = 2
            elif re.match('10\^\d*\.', txt) or re.match('10\^-\d*\.', txt):
                state = 1

        return (state, txt, pos)

    def textFromValue(self, value):
        if value >= 10000:
            return np.format_float_scientific(value, precision=4, trim='-', 
                pad_left=False, sign=False)
        else:
            return str(value)

    def valueFromText(self, text):
        if '10^' in text:
            text = text.split('^')[1]
        if text == '':
            return 0
        return float(text)

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

class FileLabel(pg.LabelItem):
    '''
        Self-adjusting label for lists of files opened
    '''
    def __init__(self, labels, *args, **kwargs):
        self.labels = labels
        font = QtGui.QFont()
        metrics = QtGui.QFontMetrics(font)
        self.widths = [metrics.boundingRect(lbl).width() for lbl in labels]
        pg.LabelItem.__init__(self, '', *args, **kwargs)
        self.setAttr('justify', 'right')
        self.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
        self.setMinimumWidth

    def resizeEvent(self, ev):
        if ev is None:
            return

        # Get the width of the new bounding rect
        newSize = ev.newSize()
        windowWidth = newSize.width() * 0.75

        # Set text based on sub-label widths and bounding rect width
        if len(self.labels) == 1:
            self.setText(self.labels[0])
        elif len(self.labels) > 1:
            # Make sure the last label is included
            lastLabel = self.labels[-1]
            estWidth = self.widths[-1]

            # Gather all labels in beginning that fit in window
            visibleLabels = []
            for lbl, width in zip(self.labels[:-1], self.widths):
                # If adding this label makes the labelItem too lengthy, break
                if len(visibleLabels) > 0 and (estWidth + width) > windowWidth:
                    break

                visibleLabels.append(lbl)
                estWidth += width

            # Build text from visibleLabels and lastLabel
            if len(visibleLabels) == (len(self.labels) - 1):
                txt = ', '.join(visibleLabels + [lastLabel])
            else:
                txt = ', '.join(visibleLabels)
                txt = ', '.join([txt, '...', lastLabel])

            self.setText(txt)

        # Default resize event
        pg.LabelItem.resizeEvent(self, ev)

class PlotGrid(GraphicsLayout):
    def __init__(self, window=None, *args, **kwargs):
        self.window = window
        self.numPlots = 0
        self.plotItems = []
        self.labels = []

        # Indicates whether label sizes should be updated when resizing
        self.labelSizesLocked = False

        # Additional lists for handling color plots
        self.colorPltKws = []
        self.colorPltInfo = {}

        # Elements used if additional tick labels are added
        self.labelSetGrd = None
        self.labelSetLabel = None
        self.labelSetLoc = 'top' # Bottom or top position
        self.startRow = 1
        self.factors = []

        super().__init__(*args, **kwargs)
        self.layout.setHorizontalSpacing(2)
        self.layout.setVerticalSpacing(2)
        self.layout.setContentsMargins(10,0,0,0)
        self.layout.setColumnStretchFactor(0, 0)
        self.layout.setRowStretchFactor(0, 0)

    def count(self):
        # Returns number of plots
        return self.numPlots

    def setHeightFactors(self, heightFactors):
        self.factors = heightFactors

    def linkTicks(self):
        if self.plotItems == []:
            return
        
        # Hide tick values/labels for all other plots and disconnect
        # signals from update function
        for plt in self.plotItems[:-1]:
            self.hideTickValues(plt)
            ba = plt.getAxis('bottom')
            if ba.receivers(ba.ticksChanged) > 0:
                ba.ticksChanged.disconnect(self.updateLabelSets)

        # Link signal from bottom axis to update func
        bottomAxis = self.plotItems[-1].getAxis('bottom')
        bottomAxis.setStyle(showValues=True)
        bottomAxis.setLabelVisible(True)
        bottomAxis.ticksChanged.connect(self.updateLabelSets)

    def setTimeLabel(self):
        if self.plotItems == []:
            return

        # Link/unlink tick changes for bottom axes and set
        # bottom axis label visible
        self.linkTicks()

        # If label sets are enabled on bottom, set the text for the
        # spacer/label item next to the axis, and hide the original
        # axis label
        self.updateTimeSpacerLabel()

    def updateLabelSets(self, ticks):
        if self.labelSetGrd:
            self.labelSetGrd.setTickLevels(ticks)
        
        self.updateTimeSpacerLabel()

    def updateTimeSpacerLabel(self):
        ''' Updates time spacer label if labelSetGrid is visible
            or resets to empty string otherwise
        '''
        if len(self.plotItems) == 0:
            return
        
        # Get spacer label item and bottom axis
        spacerLbl = self.getSideTimeLabel()
        ax = self.plotItems[-1].getAxis('bottom')

        # If in bottom axis mode, set spacer text to axis label
        if self.labelSetGrd and self.labelSetLoc == 'bottom':
            # Hide label for bottom axis
            ax.setLabelVisible(False)

            # Set spacer label to bottom axis label format
            if spacerLbl is not None:
                spacerLbl.setText(ax.get_label())
        # Reset to empty string otherwise
        elif spacerLbl is not None:
            spacerLbl.setText('')

        # Adjust layout row minimum/maximum height
        botmHt = ax.maximumHeight()
        self.layout.setRowMaximumHeight(self.startRow+self.numPlots, botmHt)
        self.layout.setRowMinimumHeight(self.startRow+self.numPlots, botmHt)

    def getSideTimeLabel(self):
        # If label sets are enabled on bottom axis, the label item
        # previously used as a spacer has its text set accordingly
        index = self.numPlots
        spacerLbl = self.layout.itemAt(self.startRow + index, 0)
        return spacerLbl

    def lockLabelSizes(self, val=True):
        self.labelSizesLocked = val

    def moveLabelSets(self, loc=None):
        # Update the label set location
        if loc is None:
            loc = self.labelSetLoc
        else:
            loc = loc.lower()
            self.labelSetLoc = loc

        if self.labelSetGrd is None:
            return

        # Remove previous label set label and grid
        self.layout.removeItem(self.labelSetLabel)
        self.layout.removeItem(self.labelSetGrd)

        # Determine label set row number
        if loc == 'top':
            row = 0
        else:
            row = self.numPlots + self.startRow + 2
            self.labelSetLabel.layout.setContentsMargins(0, 0, 0, 0)
            self.labelSetGrd.layout.setContentsMargins(0, 0, 0, 0)

        # Add label set label + grid back into layout
        self.layout.addItem(self.labelSetLabel, row, 0, 1, 1)
        self.layout.addItem(self.labelSetGrd, row, 1, 1, 1)

        # Update time label to match
        self.setTimeLabel()

    def hideTickValues(self, plt):
        bottomAxis = plt.getAxis('bottom')
        bottomAxis.showLabel(False)
        bottomAxis.setStyle(showValues=False)

    def numColorPlts(self):
        return len(self.colorPltInfo.keys())

    def resizeEvent(self, event):
        if self.numPlots == 0:
            super().resizeEvent(event)
            return

        try:
            # Fill in factors w/ 1's to make sure # of factors = # of plots
            while len(self.factors) < self.count():
                self.factors.append(1)

            # Set vertical stretch factors for each row
            for row in range(self.startRow, self.startRow + self.count()):
                self.layout.setRowStretchFactor(row, self.factors[row-self.startRow])

            # Get the plot grid height / width
            newSize = event.newSize() if event is not None else self.boundingRect()
            width, height = newSize.width(), newSize.height()

            # Estimate height for each plot
            spacingTotal = self.layout.verticalSpacing() * (len(self.plotItems) - 1)
            dateHeight = self.plotItems[-1].getAxis('bottom').height()
            if self.labelSetGrd:
                labelGrdHt = sum([lblSet.height() for lblSet in self.labelSetGrd.labelSets])
            else:
                labelGrdHt = 0
            height = height - dateHeight - labelGrdHt - spacingTotal
            numPlots = len(self.plotItems)
            totalFactors = sum(self.factors)
            plotRatios = [self.factors[i] / totalFactors for i in range(0, numPlots)]
            plotHeights = [height * ratio for ratio in plotRatios]

            # Set preferred row heights so they are all even
            for i in range(self.startRow, self.startRow + self.count()):
                pltHeight = plotHeights[i - self.startRow]
                self.layout.setRowPreferredHeight(i, pltHeight)

            # Make labels approx 1/15th of plot width
            lblWidth = max(int(width)/15, 50)

            # Resize plot labels + grad legend labels
            self.resizeLabels(lblWidth, plotHeights)

            # Adjust axis widths
            self.adjustPlotWidths()

            if event:
                event.accept()
        except:
            return

    def resizeLabels(self, lblWidth, plotHeights):
        # Find the smallest estimated font size among all the left plot labels
        fontSize = None
        for i in range(0, len(self.labels)):
            plotHeight = plotHeights[i]
            lblFontSize = self.labels[i].estimateFontSize(lblWidth, plotHeight)

            if fontSize is None:
                fontSize = lblFontSize
            fontSize = min(lblFontSize, fontSize)

        # Set the font size for all left plot labels to the minimum found above
        if not self.labelSizesLocked:
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
        fontSize = min(16, fontSize)

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
        plt.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))

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
        for axis in ['left', 'bottom', 'top', 'right']:
            plt.getAxis(axis).setStyle(tickLength=-8)
            plt.showAxis(axis)

        for ax in ['top', 'right']:
            plt.getAxis(ax).setStyle(showValues=False)
        
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
        self.moveLabelSets()

    def addColorPlt(self, plt, lblStr, colorBar, colorLbl=None, units=None,
                    colorLblSpan=1, index=None):
        if isinstance(lblStr, str):
            lbl = [lblStr]
        else:
            lbl = lblStr
            lblStr = ' '.join(lblStr)

        lbl = StackedLabel(lbl, ['#000000']*len(lbl), units=units)
        colorBar.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))
        colorBar.setMaximumHeight(1e28)
        colorBar.setOffsets(1, 1, 5, 0)
        self.addPlt(plt, lbl, index)
        index = self.numPlots - 1 if index is None else index
        self.addItem(colorBar, self.startRow + index, 2, 1, 1)
        if colorLbl:
            self.addItem(colorLbl, self.startRow + index, 3, 1, 1)

        # Additional state updates
        if self.window:
            plt.getAxis('bottom').tickOffset = self.window.tickOffset
            plt.getAxis('top').tickOffset = self.window.tickOffset
        colorBar.setBarWidth(28)

        # Add tracker lines to plots
        trackerLine = pg.InfiniteLine(movable=False, angle=90, pos=0, pen=pg.mkPen('#000000', width=1, style=QtCore.Qt.DashLine))
        if self.window:
            plt.addItem(trackerLine)
            self.window.trackerLines.insert(index, trackerLine)

        # Update state information
        cpInfo = {}
        cpInfo['Legend'] = colorBar
        cpInfo['LegendLbl'] = colorLbl
        cpInfo['Units'] = units
        self.colorPltInfo[lblStr] = cpInfo
        self.colorPltKws[index] = lblStr

    def addSpectrogram(self, specData):
        from .dynBase import SpectrogramPlotItem
        from .MagPy4 import MagPyViewBox
        # Create plot and fill with specData
        vb = MagPyViewBox(self.window, len(self.plotItems))
        plt = SpectrogramPlotItem(self.window.epoch, vb=vb)
        plt.loadPlot(specData)

        # Disable plotappr menu
        plt.setPlotMenuEnabled(False)

        # Set limits
        low, hi = self.window.minTime, self.window.maxTime
        low = low - self.window.tickOffset
        hi = hi - self.window.tickOffset
        plt.setLimits(xMin=low, xMax=hi)

        # Get color bar and axis labels
        color_bar = plt.getGradLegend(specData.log_color_scale())
        color_bar.setPlot(plt)
        color_bar.enableMenu()

        color_lbl = specData.get_legend_label()
        if specData.log_color_scale() and (not color_lbl[0].startswith('Log')):
            color_lbl[0] = f'Log {color_lbl[0]}'
        color_lbl = StackedAxisLabel(color_lbl, angle=90)
        color_lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))
        color_bar.setLabel(color_lbl)

        # Set plot labels
        y_labels = specData.get_y_label()
        axis = plt.getAxis('left')
        axis.setLabel(' '.join(y_labels.split('\n')))

        # Determine plot title
        name = specData.get_name()
        if len(name) > 10:
            # Split by spaces, not within parentheses
            match = '(' in name or ')' in name
            if '(' in name or ')' in name:
                # Find parenthesized section and split elsewhere
                match = re.search('\(.+\)', name)
                if match:
                    a, b = match.span()
                    split_name = name[:a].split(' ') + [name[a:b]] + name[b:].split(' ')
                    name = [elem for elem in split_name if elem != '']

            # Split by spaces if no parentheses detected
            if not match:
                name = name.split(' ')

        # Add elements to grid
        self.addColorPlt(plt, name, color_bar, color_lbl)
        self.resizeEvent(None)

        return plt

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
            ht = self.labelSetGrd.labelSets[0].maximumHeight()
            lbl.setMinimumHeight(ht)
            lbl.setMaximumHeight(ht)

        # Get font size from bottom date axis and apply it 
        ax = self.plotItems[-1].getAxis('bottom')
        font = QtGui.QFont() if ax.getTickFont() is None else ax.getTickFont()
        fontSize = font.pointSize()
        self.setLabelSetFontSize(fontSize)

        # Update label set location and plot widths
        self.moveLabelSets()
        self.adjustPlotWidths()

        ba = self.plotItems[-1].getAxis('bottom')
        if ba.levels is not None:
            self.updateLabelSets(ba.levels)

    def setLabelSetFontSize(self, fontSize):
        # Set font size for both label and axis item
        self.labelSetGrd.setFontSize(fontSize)
        self.labelSetLabel.setFontSize(fontSize)

        # Adjust label heights so they are aligned w/ axis items
        for lbl, ax in zip(self.labelSetLabel.subLabels, self.labelSetGrd.labelSets):
            ht = ax.maximumHeight()
            lbl.setMinimumHeight(ht)
            lbl.setMaximumHeight(ht)

        # Set font size for time label
        lbl = self.getSideTimeLabel()
        lbl.setAttr('size', f'{fontSize}pt')
        lbl.setText(lbl.text)

        ## Adjust time label height
        ht = self.plotItems[-1].getAxis('bottom').maximumHeight()
        lbl.setMinimumHeight(ht)
        lbl.setMaximumHeight(ht)

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
            else:
                # Reset time label location if all removed
                self.setTimeLabel()

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

    def adjustTitleColors(self, lineInfo):
        ''' Updates StackedLabel for the plot that the updated line is in '''
        # Extract line information
        plotIndex = lineInfo['plotIndex']
        lineIndex = lineInfo['traceIndex']
        pen = lineInfo['pen']

        # Get stacked label and set the color for the corresponding sublabel
        label = self.labels[plotIndex]
        sublabel = label.subLabels[lineIndex]
        sublabel.setColor(pen.color().name())

    def setPlotLabel(self, lbl, plotNum):
        prevLabel = self.getPlotLabel(plotNum)
        self.removeItem(prevLabel)
        self.addItem(lbl, self.startRow+plotNum, 0, 1, 1)
        self.labels[plotNum] = lbl

    def getPlotLabel(self, plotNum):
        return self.labels[plotNum]

    def setLabelFontSizes(self, val):
        for lbl in self.labels:
            lbl.setFontSize(val)

class MainPlotGrid(PlotGrid):
    def __init__(self, window=None, *args, **kwargs):
        PlotGrid.__init__(self, window, *args, **kwargs)
        self.menu = QtGui.QMenu()
        self.menu.addAction(self.window.ui.plotApprAction) # Plot appearance
        self.menu.addAction(self.window.ui.addTickLblsAction) # Additional labels

    def addPlt(self, plt, lbl, index=None):
        super().addPlt(plt, lbl, index)

        # Clicking on axes opens plot appearance
        plotFunc = self.window.openPlotAppr
        plotFunc = functools.partial(plotFunc, 1)
        for ax in ['left', 'bottom']:
            # Connect signal to plot appearance function
            bar = plt.getAxis(ax)
            bar.axisClicked.connect(plotFunc)

            # Set cursor
            bar.setCursor(QtCore.Qt.PointingHandCursor)

    def menuEnabled(self):
        return True

    def getContextMenus(self, event):
        return self.menu.actions() if self.menuEnabled() else []

    def enablePDFClipping(self, val):
        for plt in self.plotItems:
            plt.setClipToView(val)

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
    def __init__(self, dstrs, colors, units=None, html=False, *args, **kwargs):
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

    def getPens(self):
        if self.units:
            colors = self.colors[:-1]
        else:
            colors = self.colors

        pens = [pg.mkPen(color) for color in colors]
        return pens

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
        fontSize = QtGui.QFont().pointSize()
        font.setPointSize(fontSize)
        fontMetrics = QtGui.QFontMetricsF(font)

        # Estimate the current height/width of the text
        labelRects = [fontMetrics.boundingRect(lbl) for lbl in self.dstrs]
        estHeight = sum([rect.height()+2 for rect in labelRects])

        # Estimate the font size using the ratio of the rect width/height and
        # the estimated width/height 
        ratio = estHeight / (height*.75)
        newFontSize = int(min(max(2, fontSize / ratio), 16))

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

    def setHtml(self, *args):
        return

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
        self.layout.setColumnAlignment(0, QtCore.Qt.AlignTop)

    def addSubLabel(self, dstr, color):
        # Add another sublabel at end of stackedLabel
        lbl = self.addLabel(text=dstr, color=color, row=len(self.subLabels), col=0)

        # Update all internal lists
        self.subLabels.append(lbl)
        self.dstrs.append(dstr)
        self.colors.append(color)

    def setFontSize(self, pt):
        for label in self.subLabels:
            label.setText(label.text, size=f'{pt}pt')

# Checks if libraries required to use MMS Orbit tool are installed
def checkForOrbitLibs():
    installed = True
    pkgs = ['requests', 'cdflib']
    for pkg in pkgs:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            installed = False

    return installed

class ScrollSelector(QtWidgets.QAbstractSlider):
    ''' Dual slider that also serves as a scrollbar '''
    startChanged = QtCore.pyqtSignal(int)
    endChanged = QtCore.pyqtSignal(int)
    rangeChanged = QtCore.pyqtSignal(tuple)

    def __init__(self):
        # Slider start/end marker positions, max # of ticks
        self.start_pos = 0
        self.end_pos = 100
        self.num_ticks = 100

        # Keeps track of active slider
        self.current_slider = 0

        # Positions of start/end markers relative to position
        # selected on scroll bar
        self.slider_relative_start = 0
        self.slider_relative_end = 0

        # Appearance settings
        self.mark_width = 6
        self.vert_ofst = 4
        self.max_height = 25
        self.elems = {}
        self.setup_colors()

        QtWidgets.QAbstractSlider.__init__(self)
        self.setMaximumHeight(self.max_height)

    def setup_colors(self):
        # Set up gradient to use as brush for 'background' scroll area
        gradient = QtGui.QLinearGradient(0, 0, 0, 1)
        center_color = pg.mkColor((218.0, 218.0, 218.0))
        edge_color = center_color.darker(110)
        gradient.setColorAt(0, edge_color)
        gradient.setColorAt(0.25, center_color)
        gradient.setColorAt(0.75, center_color)
        gradient.setColorAt(1.0, edge_color)

        self.elems['background'] = gradient
        self.elems['outline'] = edge_color.darker(120)

        # Set up scroll bar brushes (both enabled/disabled)
        scrollColors = [pg.mkColor('#61b5ff'), pg.mkColor('#d9d9d9')]
        labels = ['scroll', 'scroll_disabled']
        for scrollColor, name in zip(scrollColors, labels):
            edgeColor = scrollColor.darker(115)
            gradient = QtGui.QLinearGradient(0, 0, 0, 0)
            gradient.setColorAt(0, edgeColor)
            gradient.setColorAt(0.1, scrollColor)
            gradient.setColorAt(0.9, scrollColor)
            gradient.setColorAt(1.0, edgeColor)
            self.elems[name] = gradient
            self.elems[f'{name}_line'] = edgeColor

        # Set the button brush
        btn_color = pg.mkColor(250, 250, 250)
        btn_grad = QtGui.QRadialGradient(0, 0, self.mark_width*2)
        btn_grad.setColorAt(0, btn_color)
        btn_grad.setColorAt(0.5, btn_color)
        btn_grad.setColorAt(1, btn_color.darker(108))
        self.elems['button'] = btn_grad

    def set_start(self, t):
        ''' Sets the start tick value '''
        t = min(max(t, 0), self.get_num_ticks())
        self.start_pos = t
        self.update()
        self.startChanged.emit(t)

    def set_end(self, t):
        ''' Sets the end tick value '''
        t = min(max(t, 0), self.get_num_ticks())
        self.end_pos = t
        self.update()
        self.endChanged.emit(t)

    def set_range(self, t):
        ''' Sets the maximum number of ticks '''
        self.num_ticks = t
        self.update()

    def get_num_ticks(self):
        ''' Returns the maximum number of ticks '''
        return self.num_ticks

    def mouseReleaseEvent(self, ev):
        self.sliderReleased.emit()

    def mouseMoveEvent(self, ev):
        if not self.isEnabled():
            return

        # Get position and current slider rects
        pos = ev.pos()
        startRect, endRect, scrollRect = self.get_marker_rects()

        # Update position of currently selected slider
        if self.current_slider == 0: # Start slider
            self.start_pos = self.get_marker_pos(pos)
            self.startChanged.emit(self.start_pos)
        elif self.current_slider == 1: # End slider
            self.end_pos = self.get_marker_pos(pos)
            self.endChanged.emit(self.end_pos)
        elif self.current_slider == -1: # Scroll bar
            self.moveBar(pos.x())

        # Update appearance
        self.update()
        ev.accept()

    def mousePressEvent(self, ev):
        # Get position and current slider rects
        pos = ev.pos()
        startRect, endRect, scrollRect = self.get_marker_rects()

        # Update which slider item is being selected 
        if startRect.contains(pos): # Start slider
            self.current_slider = 0
        elif endRect.contains(pos): # End slider
            self.current_slider = 1
        elif scrollRect.contains(pos): # Scroll bar
            self.current_slider = -1
            # Save position of start/end relative to position selected
            # on scroll bar
            self.slider_relative_start = pos.x() - scrollRect.left()
            self.slider_relative_end = scrollRect.right() - pos.x()

        else: # No slider selected
            self.current_slider = -2

    def moveBar(self, x_center):
        ''' Moves bar so that it's centered at x_center but maintains
            the initial width 
        '''
        # Get original difference between start/end
        diff = self.end_pos - self.start_pos

        # Get potential new positions of start/end sliders based
        # on new position for slider (relative to where it was selected)
        start = QtCore.QPointF(x_center - self.slider_relative_start, 0)
        end = QtCore.QPointF(x_center + self.slider_relative_end, 0)

        # Map new positions in pixel coordinates to ticks
        self.start_pos = self.get_marker_pos(start)
        self.end_pos = self.get_marker_pos(end)

        # If the new positions would shrink the scrollbar
        # set to lowest/highest values possible while still maintaining
        # the original size of the scroll bar
        if abs(self.end_pos - self.start_pos) < abs(diff):
            num_ticks = self.get_num_ticks()
            if self.end_pos == 0 or self.start_pos == 0:
                # Bounded by min value
                self.start_pos = 0
                self.end_pos = abs(diff)
            elif self.end_pos == num_ticks or self.start_pos == num_ticks:
                # Bounded by max value
                self.start_pos = num_ticks - abs(diff)
                self.end_pos = num_ticks

        # Send signal that whole range was updated
        self.rangeChanged.emit((self.start_pos, self.end_pos))

    def wheelEvent(self, ev):
        # Get angle delta
        delta = ev.angleDelta()
        if delta.y() != 0:
            diff = delta.y()
        else:
            diff = delta.x()

        # Map delta to fraction of window width
        angle = diff * (1/8)
        frac = angle / 360

        # Determine bar center
        a, b = self.get_marker_draw_pos()
        a, b = min(a, b), max(a, b)
        center = (a+b)/2

        # Save relative slider positions (to maintain width)
        self.slider_relative_start = center - a
        self.slider_relative_end = b - center

        # Move scroll bar to new center
        new_center = center + abs(b-a)*frac
        self.moveBar(new_center)
        self.update()

    def get_marker_pos(self, pos):
        ''' Maps pos's x value in pixels to an internal tick value '''
        x = pos.x()

        # Get overall area width and subtract padding for marker items
        rect = self.rect()
        width = rect.width()
        avail_width = width - (self.mark_width*2)

        # Convert x position into a tick value based on the overall
        # available space for placing ticks
        int_pos = (x - self.mark_width) * (self.get_num_ticks() / avail_width)
        int_pos = min(max(int_pos, 0), self.get_num_ticks())
        return np.round(int_pos)

    def get_marker_rects(self):
        ''' Gets the rect for each marker and scrollbar '''
        # Get rect for this object
        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # Get position of start/end markers in pixel coordinates
        avail_width = width - (self.mark_width*2) - 2
        pos_a = self.mark_width + avail_width * (self.start_pos / self.get_num_ticks())
        pos_b = self.mark_width + avail_width * (self.end_pos / self.get_num_ticks())

        # Create rects from positions and self.mark_width settings
        startRect = QtCore.QRectF(pos_a - self.mark_width, 0, self.mark_width*2, height)
        endRect = QtCore.QRectF(pos_b - self.mark_width, 0, self.mark_width*2, height)

        # Construct scrollRect from startRect and endRect endpoints
        left = startRect.left()
        right = endRect.left()
        scrollRect = QtCore.QRectF(left, 0, right-left, self.rect().height())

        return startRect, endRect, scrollRect

    def get_marker_draw_pos(self):
        # Get current rect width/height
        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # Get position of start/end markers
        avail_width = width - (self.mark_width*2)
        pos_a = self.mark_width + avail_width * (self.start_pos / self.get_num_ticks())
        pos_b = self.mark_width + avail_width * (self.end_pos / self.get_num_ticks())
        return (pos_a, pos_b)

    def paintEvent(self, ev):
        # Start a painter
        p = QtGui.QPainter()
        p.begin(self)

        # Get current rect width/height
        rect = self.rect()
        width = rect.width()
        height = rect.height()

        # Get position of start/end markers
        avail_width = width - (self.mark_width*2)
        pos_a = self.mark_width + avail_width * (self.start_pos / self.get_num_ticks())
        pos_b = self.mark_width + avail_width * (self.end_pos / self.get_num_ticks())

        # Fill in background
        rect = QtCore.QRectF(0, self.vert_ofst, rect.width()-1, height-self.vert_ofst*2)

        ## Update gradient positioning
        gradient = self.elems['background']
        gradient.setStart(0, self.vert_ofst)
        gradient.setFinalStop(0, height-self.vert_ofst)

        ## Fill with painter
        p.setBrush(gradient)
        p.setPen(self.elems['outline'])
        p.drawRoundedRect(rect, 2, 2)

        # Fill in scroll bar
        scrollRect = QtCore.QRectF(pos_a, self.vert_ofst, pos_b-pos_a, height-self.vert_ofst*2)

        ## Update gradient positioning
        enabled = self.isEnabled()
        scrollPen = self.elems['scroll_line'] if enabled else self.elems['scroll_disabled_line']
        gradient = self.elems['scroll'] if enabled else self.elems['scroll_disabled']
        gradient.setStart(0, self.vert_ofst)
        gradient.setFinalStop(0, height-self.vert_ofst)

        ## Fill with painter
        p.setBrush(gradient)
        p.setPen(scrollPen)
        p.drawRect(scrollRect)

        # Fill in markers
        startRect = QtCore.QRectF(pos_a - self.mark_width, 1, self.mark_width*2, height-2)
        endRect = QtCore.QRectF(pos_b - self.mark_width, 1, self.mark_width*2, height-2)

        ## Set up pen and aliasing
        pen = pg.mkPen(self.elems['outline'].darker(120))
        p.setPen(pen)
        p.setRenderHint(p.Antialiasing, True)

        ## Update gradient positions and draw
        for rect in [startRect, endRect]:
            gradient = QtGui.QRadialGradient(rect.center(), self.mark_width*2)
            gradient.setStops(self.elems['button'].stops())
            p.setBrush(gradient)
            p.drawRoundedRect(rect, 1.0, 1.0)

        p.end()