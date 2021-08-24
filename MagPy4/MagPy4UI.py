
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
import importlib.util

import numpy as np
import pyqtgraph as pg
import functools
import os

from .plot_extensions import GridGraphicsLayout
from . import getRelPath
from .mth import Mth
import re
from . import config

from enum import Enum

class ScrollArea(QtWidgets.QScrollArea):
    ''' ScrollArea with a adjustToWidget function that
        sets an appropriate minimum width / height for
        the widget
    '''
    def __init__(self):
        self.preferred_size = (300, 500)
        super().__init__()
    
    def setPreferredSize(self, width, height):
        ''' Sets the preferred widget size if the sizeHint is not valid '''
        self.preferred_size = (width, height)
    
    def adjustToWidget(self):
        ''' Adjusts ScrollArea size to sizeHint of widget if valid,
            otherwise it is set to preferred_size
        '''
        # Get widget sizeHint
        widget = self.widget()
        min_width = widget.sizeHint().width()
        min_height = widget.sizeHint().height()

        # Check widget size validity
        if min_width < 25 or min_width > 750:
            min_width = self.preferred_size[0]
        if min_height < 25 or min_height > 800:
            min_height = self.preferred_size[1]

        # Set minimum width and height for the ScrollArea
        self.setMinimumWidth(min_width + 20)
        self.setMinimumHeight(min_height + 20)

    def scrollToRow(self, row):
        ''' Scrolls to the widget's given row '''
        pos = self.widget().getRowPosition(row)
        self.verticalScrollBar().setValue(pos)

class FileLoadDialog(QtWidgets.QDialog):
    ''' Subclassed QDialog mimics QProgressDialog but allows
        for an additional widget / layout to be displayed between
        the label and the progress bar
    '''
    canceled = QtCore.pyqtSignal()
    def __init__(self, *args, **kwargs):
        self._cancelled = False # Whether task has been cancelled or not
        super().__init__(*args, **kwargs)

        # Create widgets
        self.label = QtWidgets.QLabel()
        self.widget = None
        self.progbar = QtWidgets.QProgressBar()
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.cancel)

        # Add items to layout
        self._layout = QtWidgets.QGridLayout()
        self.setLayout(self._layout)

        cancel_layout = QtWidgets.QHBoxLayout()
        cancel_layout.addStretch()
        cancel_layout.addWidget(self.cancel_btn)

        self._layout.addWidget(self.label, 0, 0, 1, 1)
        self._layout.addWidget(self.progbar, 2, 0, 1, 1)
        self._layout.addLayout(cancel_layout, 3, 0, 1, 1)

        # Connect reject to cancel
        self.rejected.connect(self._internal_cancel)

    def setWidget(self, widget, show=True):
        ''' Sets the widget between the label text and the progress bar '''
        # Remove old widget
        item = self._layout.itemAtPosition(1, 0)
        if item is not None:
            self._layout.removeItem(item)
        
        # Add widget to layout and save
        self._layout.addWidget(widget, 1, 0, 1, 1)
        self.widget = widget
        self.widget.setVisible(show)
    
    def getWidget(self):
        ''' Get the widget set between the label and progress bar '''
        return self.widget
    
    def setBar(self, bar):
        ''' Sets the progress bar used '''
        # Remove old progress bar
        item = self._layout.itemAtPosition(2, 0)
        self._layout.removeItem(item)
        item.widget().deleteLater()

        # Add progress bar to layout and save
        self._layout.addWidget(bar, 2, 0, 1, 1)
        self.progbar = bar
    
    def getBar(self):
        ''' Gets the progress bar widget '''
        return self.progbar

    def setLabelText(self, txt):
        ''' Sets the label text for the dialog '''
        self.label.setText(txt)
    
    def setMinimum(self, val):
        ''' Sets the progress bar minimum '''
        self.progbar.setMinimum(val)
    
    def setMaximum(self, val):
        ''' Sets the progress bar maximum '''
        self.progbar.setMaximum(val)
    
    def setValue(self, val):
        ''' Set the progress bar value '''
        self.progbar.setValue(val)
    
    def cancel(self):
        ''' Signal to cancel the task '''
        self.reject()
        self._internal_cancel()
        self._cancelled = True
    
    def wasCanceled(self):
        ''' Checks if task has been cancelled '''
        return self._cancelled

    def _internal_cancel(self):
        self.canceled.emit()

class ProgStates(Enum):
    ''' File load states '''
    LOADING = 0
    SUCCESS = 1
    FAILURE = -1

class IconBtn(QtWidgets.QPushButton):
    ''' QLabel that displays a small image / icon only '''
    def __init__(self, img_path=None):
        self.data = None
        self.img_size = (25, 25)
        self.img_path = None
        super().__init__()
        self.setFlat(True)
        if img_path is not None:
            self.setImage(img_path)

    def setImage(self, img_path):
        ''' Sets the pixmap to the image at the given path '''
        self.img_path = img_path
        image = QtGui.QPixmap(img_path)
        icon = QtGui.QIcon(image)
        self.setIcon(icon)
    
    def setImageSize(self, width, height):
        ''' Sets the image to the scaled size '''
        self.img_size = (width, height)
        self.setIconSize(QtCore.QSize(width, height))

    def _setIcon(self, pixmap):
        icon = QtGui.QIcon(pixmap)
        self.setIcon(icon)

    def setData(self, data):
        ''' Sets any user data '''
        self.data = data
    
    def getData(self):
        ''' Returns any user data '''
        return self.data

class ProgressChecklist(QtWidgets.QFrame):
    ''' Widget that displays a list of items and a small icon next
        to each of them that indicates whether it has been successfull,
        failed, or is the current task
    '''
    img_path = getRelPath('images')
    image_dict = {
        ProgStates.LOADING : os.path.join(img_path, 'l_loading_icon.svg'),
        ProgStates.SUCCESS : os.path.join(img_path, 'finished_icon.svg'),
        ProgStates.FAILURE : os.path.join(img_path, 'failure_icon.svg'),
    }
    def __init__(self, items=[]):
        # Set up widget
        super().__init__()
        self.layout = QtWidgets.QGridLayout()
        self.layout.setContentsMargins(5, 0, 5, 0)
        self.layout.setColumnStretch(1, 0)
        self.setLayout(self.layout)

        # Set up any items in checklist
        if len(items) > 0:
            self.setItems(items)
    
    def addItem(self, name):
        # Adds item to end of list
        r = self.layout.rowCount()
        if self.layout.count() == 0:
            r = 0

        # Create label
        font = QtGui.QFont()
        font.setPointSize(12)

        label = QtWidgets.QLabel(name)
        label.setFont(font)

        # Create status icon
        h, w = 22, 22
        btn = IconBtn()
        btn.setMaximumWidth(50)
        btn.setImageSize(h, w)
    
        # Add label andIconBtn icon to last row of layout
        self.layout.addWidget(label, r, 0, 1, 1)
        self.layout.addWidget(btn, r, 1, 1, 1)
    
    def setItems(self, items):
        ''' Initializes the list of items in the checklist '''
        for item in items:
            self.addItem(item)
        
    def getItems(self):
        ''' Returns the list of items in the checklist '''
        count = self.layout.rowCount()
        items = [self._getLabel(r).text() for r in range(count)]
        return items
    
    def getStatuses(self):
        ''' Returns the status of each item in the checklist '''
        count = self.layout.rowCount()
        values = [self._getBtn(r).getData() for r in range(count)]
        return values

    def getFailures(self):
        ''' Returns the items that have a 'FAILURE' status '''
        items = self.getItems()
        statuses = self.getStatuses()
        failures = [item for item, s in zip(items, statuses) if s == ProgStates.FAILURE]
        return failures
    
    def _getBtn(self, row):
        item = self.layout.itemAtPosition(row, 1)
        return item.widget()
    
    def _getLabel(self, row):
        item = self.layout.itemAtPosition(row, 0)
        return item.widget()

    def setStatus(self, row, value):
        ''' Sets the status for the item in the given row to value '''
        # Skip if not a valid value
        if value not in self.image_dict:
            return

        # Get button and set data and icon
        btn = self._getBtn(row)
        url = self.image_dict[value]
        btn.setData(value)
        btn.setImage(url)

        # Highlight current task being updated
        if value == ProgStates.LOADING:
            self.highlightRow(row, True)
        else:
            self.highlightRow(row, False)
    
    def highlightRow(self, row, val=True):
        ''' Bolds the text label for the item in the given row '''
        label = self._getLabel(row)
        font = label.font()
        font.setBold(val)
        label.setFont(font)
    
    def getRowPosition(self, row):
        ''' Gets the relative position of the given row in widget
            coordinates
        '''
        item = self._getLabel(max(0, row - 3))
        pt = item.geometry().top()
        return pt
    
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
        self.actionOpenFF.setText('Open Flat File...')
        self.actionOpenFF.setStatusTip('Opens a flat file')

        self.actionOpenCDF = QtWidgets.QAction(window)
        self.actionOpenCDF.setText('Open CDF File...')
        self.actionOpenCDF.setStatusTip('Opens a CDF file (currently experimental)')

        self.actionOpenASCII = QtWidgets.QAction(window)
        self.actionOpenASCII.setText('Open ASCII File...')
        self.actionOpenASCII.setStatusTip('Opens a simple ASCII file')

        self.actionOpenFile = QtWidgets.QAction(window)
        self.actionOpenFile.setText('&Open File...')
        self.actionOpenFile.setStatusTip('Opens a new file')
        self.actionOpenFile.setShortcut('Ctrl+O')
        
        self.actionAddFile = QtWidgets.QAction(window)
        self.actionAddFile.setText('&Add File...')
        self.actionAddFile.setStatusTip('Add a new file')
        self.actionAddFile.setShortcut('Ctrl+A')

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
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionOpenFile)
        self.fileMenu.addAction(self.actionAddFile)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionOpenFF)
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
        from .plotBase import GraphicsView
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

class TimeEdit(QtWidgets.QWidget):
    rangeChanged = QtCore.pyqtSignal(tuple)
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

        self.start.dateTimeChanged.connect(self._range_edited)
        self.end.dateTimeChanged.connect(self._range_edited)

        super().__init__()
    
    def _range_edited(self):
        start = self.start.dateTime().toPyDateTime()
        end = self.end.dateTime().toPyDateTime()
        self.rangeChanged.emit((start, end))

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
        self._range_edited()
    
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
                    w = QtWidgets.QLabel('0.0')
                    w.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                elif type == 'lines':
                    w = QtWidgets.QLineEdit()
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
        self.rangeChanged.emit((self.start_pos, self.end_pos))

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
