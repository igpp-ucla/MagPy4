"""
Main module for the program
handles data, plotting and main window management
"""

# python 3.6
import os
import sys
import pickle
import argparse
import json
from fflib import ff_reader, ff_time, ff_writer
from . import plot_helper
from . import data_util

# Version number and copyright notice displayed in the About box
from . import get_relative_path, MAGPY_VERSION, USERDATALOC

NAME = f'MagPy4'
VERSION = f'Version {MAGPY_VERSION}'
COPYRIGHT = f'Copyright © 2020 The Regents of the University of California'

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
import numpy.lib.recfunctions as rfn
import pyqtgraph as pg
from .MagPy4UI import MagPy4UI, FileLabel, ProgStates, FileLoadDialog, ProgressChecklist, ScrollArea
from .pgextensions import TrackerRegion
from .widgets.plotmenu import PlotMenu
from .widgets.spectra import Spectra
from .widgets.datadisplay import DataDisplay
from .widgets.edit import Edit
from .widgets.tracestats import TraceStats
from .widgets.helpwin import HelpWindow
from .widgets.aboutdialog import aboutdialog
from .plotbase import MagPyPlotItem
from .widgets.mmstools import PlaneNormal, Curlometer, Curvature, PressureTool, get_mms_grps
from .widgets.mms_data import MMSDataDownloader
from .widgets import mms_orbit
from .widgets import mms_formation
from .widgets.detrendwin import DetrendWindow
from .widgets.dynamicspectra import DynamicSpectra, DynamicCohPha
from .widgets.waveanalysis import DynamicWave
from .widgets.trajectory import TrajectoryAnalysis
from .mth import Mth
from scipy import interpolate
import bisect
from .selectbase import GeneralSelect, FixedSelection, TimeRegionSelector, BatchSelect
from .layouttools import LabeledProgress
from .data_import import merge_datas, find_vec_grps, get_resolution_diffs
from .data_import import load_text_file, load_flat_file, load_cdf
import numpy.lib.recfunctions as rfn
from .qtthread import TaskRunner, ThreadPool, TaskThread

from .grid import PlotGridObject
from .plotbase import StackedLabel
from . import plot_helper

import functools
import traceback
from copy import copy
from datetime import datetime
from itertools import cycle

CANREADCDFS = False

# Maps file types to extension filters in 'Open File' dialog
file_types_dict = {
    'ASCII' : 'Text File (*.txt *.tab *.csv *.dat *.tsv)',
    'FLATFILE' : 'FlatFile (*.ffd)',
    'CDF' : 'CDF (*.cdf)'
}

class MagPy4Window(QtWidgets.QMainWindow, MagPy4UI):
    def __init__(self, app, parent=None):
        super(MagPy4Window, self).__init__(parent)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True) #todo add option to toggle this

        self.app = app
        self.ui = MagPy4UI()
        self.ui.setupUI(self)

        global CANREADCDFS

        self.OS = os.name
        if os.name == 'nt':
            self.OS = 'windows'
        print(f'OS: {self.OS}')

        self.epoch = None
        self.tE = 0
        self.tO = 0
        self.minTime = 0
        self.maxTime = 0

        self.ui.scrollSelect.startChanged.connect(self.onStartSliderChanged)
        self.ui.scrollSelect.endChanged.connect(self.onEndSliderChanged)
        self.ui.scrollSelect.sliderReleased.connect(self.sliderReleased)
        self.ui.scrollSelect.rangeChanged.connect(self._scroll_updated)

        # Shift window connections
        self.ui.mvRgtBtn.clicked.connect(self.shiftWinRgt)
        self.ui.mvLftBtn.clicked.connect(self.shiftWinLft)
        self.ui.mvLftShrtct.activated.connect(self.shiftWinLft)
        self.ui.mvRgtShrtct.activated.connect(self.shiftWinRgt)
        self.ui.shftPrcntBox.valueChanged.connect(self.updtShftPrcnt)

        # Zoom window connects
        self.ui.zoomInShrtct.activated.connect(self.zoomWindowIn)
        self.ui.zoomOutShrtct.activated.connect(self.zoomWindowOut)
        self.ui.zoomAllShrtct.activated.connect(self.viewAllData)
        self.zoomFrac = 0.4

        self.ui.timeEdit.rangeChanged.connect(self._xrange_updated)

        # Scrolling zoom connects
        self.ui.scrollPlusShrtct.activated.connect(self.increasePlotHeight)
        self.ui.scrollMinusShrtct.activated.connect(self.decreasePlotHeight)

        # Main menu action connections
        self.ui.actionOpenFile.triggered.connect(self.openFileDialog)
        self.ui.actionAddFile.triggered.connect(self.addFileDialog)
        self.ui.actionOpenFF.triggered.connect(self.openFlatFile)
        self.ui.actionOpenASCII.triggered.connect(self.openASCII)
        self.ui.actionOpenCDF.triggered.connect(self.openCDF)

        self.ui.actionExportDataFile.triggered.connect(self.exportFile)

        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionOpenWs.triggered.connect(self.openWsOpenDialog)
        self.ui.actionSaveWs.triggered.connect(self.openWsSaveDialog)
        self.ui.actionHelp.triggered.connect(self.openHelp)
        self.ui.actionAbout.triggered.connect(self.openAbout)
        self.ui.switchMode.triggered.connect(self.swapMode)
        self.ui.runTests.triggered.connect(self.runTests)
        self.ui.actionBatchSelect.triggered.connect(self.openBatchSelect)
        self.ui.actionChkForUpdt.triggered.connect(self.checkForUpdate)

        # Selection menu actions
        self.ui.actionFixSelection.triggered.connect(self.fixSelection)
        self.ui.actionSelectByTime.triggered.connect(self.openTimeSelect)
        self.ui.actionSelectView.triggered.connect(self.autoSelectRange)

        # options menu dropdown
        self.ui.scaleYToCurrentTimeAction.toggled.connect(self.updateYRange)
        self.ui.antialiasAction.toggled.connect(self.toggleAntialiasing)
        self.ui.bridgeDataGaps.toggled.connect(self.replotDataCallback)
        self.ui.drawPoints.toggled.connect(self.replotDataCallback)
        self.ui.downsampleAction.toggled.connect(self.enableDownsampling)
        self.ui.enableScrollingAction.toggled.connect(self.enableScrolling)

        # Disable the Tools and Options menus. They'll be enabled after the user opens a file.
        self.DATASTRINGS = []
        self.enableToolsAndOptionsMenus(False)

        self.epoch = None
        self.errorFlag = None

        self.helpwin = None
        self.aboutDialog = None
        self.FIDs = []
        self.tickOffset = 0 # Smallest tick in data, used when plotting x data
        self.smoothing = None
        self.currSelect = None
        self.savedRegion = None
        self.fixedSelect = None
        self.timeSelect = None
        self.asc = None
        self.batchSelect = None
        self.minimumPlotHeight = 3 # Inches
        self.tracker = None

        self.savedPlotInfo = None

        # Tool full names
        self.toolNames = ['Data', 'Edit', 'Plot Menu', 'Detrend', 'Spectra',
            'Dynamic Spectra', 'Dynamic Coh/Pha', 'Wave Analysis',
            'Trajectory Analysis', 'Stats', 'Plane Normal', 'Curlometer', 
            'Curvature', 'MMS Orbit', 'MMSFormation', 'MMS Pressure', 'MMSData']

        # Internal tool abbreviation
        self.toolAbbrv = ['Data', 'Edit', 'PlotMenu', 'Detrend', 'Spectra',
            'DynSpectra', 'DynCohPha', 'DynWave', 'Traj', 'Stats', 'PlaneNormal', 'Curlometer', 
            'Curvature', 'MMSOrbit', 'MMSFormation', 'Pressure', 'MMSData']
        
        # Tools that are 'selectable'
        self.select_opts = self.toolAbbrv[3:13] + [self.toolAbbrv[15]]

        # Tool actions to trigger
        self.toolActions = [self.ui.actionShowData, self.ui.actionEdit,
            self.ui.actionPlotMenu, self.ui.actionDetrend, self.ui.actionSpectra,
            self.ui.actionDynamicSpectra, self.ui.actionDynamicCohPha, self.ui.actionDynWave,
            self.ui.actionTraj, self.ui.actionTraceStats, self.ui.actionPlaneNormal,
            self.ui.actionCurlometer, self.ui.actionCurvature, self.ui.actionMMSOrbit,
            self.ui.actionMMSFormation, self.ui.actionMMSPressure, self.ui.actionLoadMMS]
        
        # Object classes for each tool
        self.widget_classes = {}
        toolClasses = [DataDisplay, Edit, PlotMenu, DetrendWindow, Spectra,
            DynamicSpectra, DynamicCohPha, DynamicWave, TrajectoryAnalysis,
            TraceStats, PlaneNormal, Curlometer, Curvature,
            mms_orbit.MMS_Orbit, mms_formation.MMS_Formation,
            PressureTool, MMSDataDownloader]

        # Functions for opening non-select tools
        self.toolInitFuncs = {
            'Data' : self.showData,
            'Edit' : self.openEdit,
            'PlotMenu' : self.openPlotMenu,
            'MMSOrbit' : self.openMMSOrbit,
            'MMSFormation' : self.openMMSFormation,
            'MMSData' : self.openMMSData,
            'PlaneNormal': self.startPlaneNormal,
        }

        # Selection colors
        colors = {
            'red' : '#a60000',
            'green' : '#59bf00',
            'aqua': '#09b8b2',
            'blue': '#0949b8',
            'purple': '#5509b8',
            'pink': '#d92b9c',
            'gold' : '#d19900',
        }

        # Selection colors given to each tool
        colors_list = cycle(list(colors.keys()))
        self.select_colors = {}
        for tool in self.toolAbbrv:
            if tool not in self.toolInitFuncs:
                color = next(colors_list)
                self.select_colors[tool] = colors[color]

        # Selection types applied to specific tools, default is 'Single'
        self.tool_select_types = {
            'Stats' : 'Adjusting',
            'Curlometer' : 'Multi',
            'Curvature' : 'Multi'
        }

        # Functions for updating specific tools
        self.tool_updt_funcs = {
            'Stats' : self.updateTraceStats,
            'DynSpectra' : self.updateDynamicSpectra,
            'DynCohPha' : self.updateDynCohPha,
            'Curlometer' : self.updateCurlometer,
            'DynWave' : self.updateDynWave
        }

        # Initialize tool values and triggers
        self.tools = {}
        self.toolNameMap = {}
        self.toolAbbrToName = {}
        for name, abbrv, act, cls in zip(self.toolNames, self.toolAbbrv, self.toolActions, toolClasses):
            # Tool attributes
            self.toolNameMap[name] = abbrv
            self.toolAbbrToName[abbrv] = name
            self.tools[abbrv] = None
            self.widget_classes[abbrv] = cls

            # Tool 'start' function
            if abbrv in self.toolInitFuncs:
                toolFunc = self.toolInitFuncs[abbrv]
            else:
                toolFunc = functools.partial(self.startTool, abbrv)

            # Connect action to start trigger
            act.triggered.connect(toolFunc)
        
        # Initialize wave analysis functions
        key = 'DynWave'
        for act in self.ui.dynWaveActions:
            text = act.text()[:-3]
            set_func = functools.partial(self.setWaveAnalysis, text)
            toolFunc = functools.partial(self.startTool, key, None, set_func)
            act.triggered.connect(toolFunc)

        # these are saves for options for program lifetime
        self.plotmenuTableMode = True
        self.traceStatsOnTop = True
        self.mouseEnabled = False

        self.mmsInterp = {}

        self.initDataStorageStructures()

        # this is where options for plot lifetime are saved
        self.initVariables()

        # Shift percentage setup
        self.shftPrcnt = self.ui.shftPrcntBox.value()/100 # Initialize default val

        # Cutoff values for time-label properties
        self.dayCutoff = 60 * 60 * 24
        self.hrCutoff = 60 * 60 * 1.5
        self.minCutoff = 10 * 60

        self.magpyIcon = QtGui.QIcon()
        self.marsIcon = QtGui.QIcon()
        img_path = get_relative_path('images', directory=True)
        if self.OS == 'mac':
            self.magpyIcon.addFile(img_path+'magPy_blue.hqx')
            self.marsIcon.addFile(img_path+'mars.hqx')
        else:
            self.magpyIcon.addFile(img_path+'magPy_blue.ico')
            self.marsIcon.addFile(img_path+'mars.ico')

        self.app.setWindowIcon(self.magpyIcon)

        # setup pens
        self.pens = []
        self.mmsColors = ['#000005', '#d55e00', '#009e73', '#56b4e9']
        # Blue, Green, Red, Yellow, Magenta, Cyan, Purple, Black
        colors = ['#0000ff','#00ad05','#ea0023', '#fc9f00', '#ff00e1', '#00ddb1',
            '#9400ff', '#191919']
        for c in colors:
            self.pens.append(pg.mkPen(c, width=1))# style=QtCore.Qt.DotLine)
        self.trackerPen = pg.mkPen('#000000', width=1, style=QtCore.Qt.DashLine)
        self.customPens = []

        self.pltGrd = None
        self.plotItems = []
        self.trackerLines = []
        self.regions = []

        self.hoverTracker = None

        self.startUp = True
        self.workspace = None
        self.pool = ThreadPool()
        self.load_dialog = None

    def _scroll_updated(self, trange):
        start, stop = trange
        start += self.minTime
        stop += self.minTime
        self._xrange_updated((start, stop))
    
    def _xrange_updated(self, trange):
        start, stop = trange
        if isinstance(start, (datetime, np.datetime64)):
            start_tick = ff_time.date_to_tick(start, self.epoch) 
            end_tick = ff_time.date_to_tick(stop, self.epoch) 
            start_date = start
            stop_date = stop
        else:
            start_tick = start
            end_tick = stop

            start_date = ff_time.tick_to_date(start, self.epoch)
            stop_date = ff_time.tick_to_date(stop, self.epoch)

        self.tO = start_tick
        self.tE = end_tick

        self.ui.scrollSelect.blockSignals(True)
        self.ui.scrollSelect.set_start(start_tick-self.minTime)
        self.ui.scrollSelect.set_end(end_tick-self.minTime)
        self.ui.scrollSelect.blockSignals(False)
        
        self.ui.timeEdit.blockSignals(True)
        self.ui.timeEdit.start.blockSignals(True)
        self.ui.timeEdit.end.blockSignals(True)

        self.ui.timeEdit.setTimeRange(start_date, stop_date)

        self.ui.timeEdit.start.blockSignals(False)
        self.ui.timeEdit.end.blockSignals(False)
        self.ui.timeEdit.blockSignals(False)
        
        self.pltGrdObject.blockSignals(True)
        self.pltGrd.set_x_range(start_tick, end_tick)
        self.pltGrdObject.blockSignals(False)

    def updateStateFile(self, key, val):
        ''' Updates state file with key value pair '''
        # Get state file and read in contents
        state_file_path = os.path.join(USERDATALOC, 'state.json')
        state_dict = self.readStateFile()

        # Update state dictionary
        state_dict[key] = val

        # Open state file for rewriting and close
        fd = open(state_file_path, 'w')
        json.dump(state_dict, fd)
        fd.close()

    def readStateFile(self):
        ''' Read in state dictionary from file '''
        state_file_path = os.path.join(USERDATALOC, 'state.json')
        state_dict = {}
        if os.path.exists(state_file_path):
            fd = open(state_file_path, 'r')
            state_dict = json.load(fd)
            fd.close()
        return state_dict

    def checkForUpdate(self):
        # Close window before updating
        self.close()
        self.app.processEvents()

        # Run update command and then re-run program
        updateMagPy()
        cmd = 'MagPy4' if os.name != 'nt' else 'MagPy4.exe'
        os.system(cmd)

    def getPenInfo(self, pen):
        color = pen.color().name()
        width = pen.width()
        style = pen.style()
        return (color, width, style)

    def makePen(self, color, width, style):
        pen = pg.mkPen(color)
        pen.setWidth(width)
        pen.setStyle(style)
        return pen

    def bringToolsToFront(self):
        for toolName, toolObj in self.tools.items():
            if toolObj:
                toolObj.raise_()

    def openBatchSelect(self, sigHand=False, closeTools=True):
        if closeTools:
            self.closeTimeSelect()
            self.closeFixSelection()
            self.closeBatchSelect()
            self.closePlotTools()

        self.batchSelect = BatchSelect(self)
        self.batchSelect.show()
        self.updateSelectionMenu()

    def closeBatchSelect(self):
        if self.batchSelect:
            self.batchSelect.close()
            self.batchSelect = None
        self.updateSelectionMenu()

    def getWinTickWidth(self):
        winWidth = abs(self.tE - self.tO) # Number of ticks currently displayed
        return winWidth

    def setNewWindowTicks(self, newTO, newTE):
        start = ff_time.tick_to_date(newTO, self.epoch)
        stop = ff_time.tick_to_date(newTE, self.epoch)
        self.ui.timeEdit.setTimeRange(start, stop)

    def shiftWindow(self, direction):
        winWidth = self.getWinTickWidth() # Number of ticks currently displayed
        shiftAmt = winWidth*self.shftPrcnt

        if direction == 'L': # Shift amt is negative if moving left
            shiftAmt = shiftAmt * (-1) 
        newTO = self.tO + shiftAmt
        newTE = self.tE + shiftAmt

        # Case where adding shift amnt gives ticks beyond min and max ticks
        if self.tO > self.tE:
            # If 'start' time > 'end' time, switch vals for comparison
            newTE, newTO = self.chkBoundaries(newTE, newTO, winWidth)
        else:
            newTO, newTE = self.chkBoundaries(newTO, newTE, winWidth)
        
        self.setNewWindowTicks(newTO, newTE)

    def shiftWinRgt(self):
        self.shiftWindow('R')

    def shiftWinLft(self):
        self.shiftWindow('L')

    def chkBoundaries(self, origin, end, winWidth):
        # When adding/subtracting shift amount goes past min and max ticks,
        # shift window to that edge while maintaining the num of ticks displayed
        if (origin < self.minTime):
            origin = self.minTime
            end = origin + winWidth
        elif end > self.maxTime:
            end = self.maxTime
            origin = end - winWidth
        return (origin, end)

    def updtShftPrcnt(self):
        self.shftPrcnt = self.ui.shftPrcntBox.value()/100

    def zoomWindowIn(self):
        self.zoomWindow(True)

    def zoomWindowOut(self):
        self.zoomWindow(False)

    def zoomWindow(self, zoomIn):
        # Get amount of ticks in window to reduce/increase width by
        winWidth = self.getWinTickWidth()
        zoomAmnt = winWidth * self.zoomFrac
        halfZoomAmnt = zoomAmnt / 2

        # Get tentative new start/end ticks
        start, end = self.iO, self.iE
        if zoomIn: # Reduce width by increasing start and decreasing end
            start = start + halfZoomAmnt
            end = end - halfZoomAmnt
        else:
            start = start - halfZoomAmnt
            end = end + halfZoomAmnt

        # Wrap endpoints and adjust window widths
        start = min(max(start, 0), self.iiE)
        end = max(min(end, self.iiE), 0)

        if end - start < 4: # Do not do anything if zooming in results in few ticks visible
            return

        self.setNewWindowTicks(start, end)

    def zoomCentered(self, ev, axis):
        ''' Distributes wheel events across plots in grid
            and updates other UI states accordingly
        '''

        # Pass scroll event to all plot items
        for plot in self.plotItems:
            vb = plot.getViewBox()
            vb.setMouseEnabled(x=True, y=False)
            pg.ViewBox.wheelEvent(vb, ev, axis)
            vb.setMouseEnabled(x=False)
        
        # Get view range, map to datetimes, and set
        # local attributes
        (xmin, xmax), yrng = vb.viewRange()
        xmin += self.tickOffset
        xmax += self.tickOffset

        start = ff_time.tick_to_date(xmin, self.epoch)
        stop = ff_time.tick_to_date(xmax, self.epoch)

        self.tO = xmin
        self.tE = xmax

        # Update time edits with new range w/o signals
        edits = [self.ui.timeEdit.start, self.ui.timeEdit.end]
        vals = [start, stop]
        for edit, val in zip(edits, vals):
            edit.blockSignals(True)
            edit.setDateTime(val)
            edit.blockSignals(False)
            edit.update()

        # Update sliders based on time edit values
        self.onStartEditChanged(start, False)
        self.onEndEditChanged(stop, False)

        # Update y range
        self.updateYRange()

    def viewAllData(self):
        self.setNewWindowTicks(0, self.iiE)

    # Use these two functions to set a temporary status msg and clear it
    def showStatusMsg(self, msg):
        status = 'STATUS: ' + msg
        self.ui.statusBar.showMessage(status)

    def clearStatusMsg(self):
        self.ui.statusBar.clearMessage()

    def enableToolsAndOptionsMenus(self, b):
        """Enable or disable the Tools and Options menus.
        """
        self.ui.toolsMenu.setEnabled(b)
        self.ui.optionsMenu.setEnabled(b)
        self.ui.selectMenu.setEnabled(b)
        self.checkForPosDta()

    def checkForPosDta(self):
        traj = TrajectoryAnalysis(self)
        if not traj.validState():
            self.ui.actionTraj.setVisible(False)
        else:
            self.ui.actionTraj.setVisible(True)

    # close any subwindows if main window is closed
    # this should also get called if flatfile changes
    def closeEvent(self, event):
        self.closeAllSubWindows()
        if self.pltGrdObject:
            self.pltGrdObject.closePlotAppr()
            self.pltGrdObject.deleteLater()

    def openWsOpenDialog(self):
        # Opens file dialog for user to select a workspace file to open
        fd = QtWidgets.QFileDialog(self)
        fd.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        fd.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        fd.setNameFilters(['MagPy Files (*.mp)', 'Any files (*)'])
        fd.fileSelected.connect(self.openWorkspace)
        fd.open()

    def openWsSaveDialog(self):
        # Opens dialog for user to select a file to save the workspace to
        fd = QtWidgets.QFileDialog(self)
        fd.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        fd.setFileMode(QtWidgets.QFileDialog.AnyFile)
        fd.setDefaultSuffix('.mp')
        fd.fileSelected.connect(self.saveWorkspace)
        fd.open()

    def openWorkspace(self, filename):
        if filename is None:
            return

        # Attempt to open workspace file and check if it is empty
        errMsg = 'Error: Could not open workspace'
        try:
            wsf = open(filename, 'rb')
        except:
            self.ui.statusBar.showMessage(errMsg, 5000)
            return

        if os.path.getsize(filename) <= 0:
            self.ui.statusBar.showMessage('Error: Workspace file is empty', 5000)
            return

        # Load the state dictionary and close the worksapce file
        winDict = pickle.load(wsf)
        wsf.close()

        # Load the correct MarsPy/MagPy mode
        mode = winDict['Mode']
        if mode != self.insightMode:
            self.insightMode = mode
            if not self.startUp: # Update UI if not opening at startup screen
                self.swapModeUI()

        # Load saved files
        savedFiles = winDict['Files']
        res = True
        try:
            res = self.loadFileState(savedFiles)
            if not res: # File loading failed
                raise
        except:
            # If file loading fails, show an error msg and stop here
            self.ui.statusBar.showMessage(errMsg, 5000)
            return

        # Load state information for main window, plot grid, and tools
        self.loadDataState(winDict['Data'])
        self.loadEditState(winDict['Edits'])
        self.loadTimeState(winDict['Time'])
        self.loadPlotState(winDict['Plots'])
        self.loadToolState(winDict['Tools'])
        self.loadSelectState(winDict['Select'])
        self.loadGridOptions(winDict['GridOptions'])

        # Make sure tools are on top of main window and save workspace filename
        QtCore.QTimer.singleShot(50, self.bringToolsToFront)
        self.workspace = filename

    def loadFileState(self, state):
        # Extract state info
        fileNames = state['Names']
        fileType = state['FileType']

        # Check if loading flat files v. ASCII files
        ffMode = True if fileType == 'FLAT FILE' else False
        if ffMode: # Add correction extension to flat file names
            fileNames = [ff+'.ffd' for ff in fileNames]

        # Initialize ASCII_Importer instance if not loading flat files and pass in
        # state info about the file format
        if not ffMode:
            self.openAscDialog(fileNames, state['Desc'])

        # Open files from state
        res = self.openFileList(fileNames, ffMode, True)
        return res # Return whether all files were successfully opened or not

    def loadDataState(self, dataDict):
        # Load state info related to DATADICT, DATASTRINGS, custom variables, etc.
        for kw in dataDict:
            setattr(self, kw, dataDict[kw])

    def loadTimeState(self, timeDict):
        self.TIMES = timeDict['TIMES']
        self.TIMEINDEX = timeDict['TIMEINDEX']
        self.resolution = timeDict['RESOLUTION']
        (minTime, maxTime), (tO, tE) = timeDict['WIN_TIMES']
        (minTick, maxTick), (iO, iE) = timeDict['WIN_TICKS']
        self.minTime = minTime
        self.maxTime = maxTime
        self.setNewWindowTicks(iO, iE)

    def saveWorkspace(self, filename):
        if filename is None: # Do nothing if no filename given
            return

        # Save the state information into a dictionary
        data = self.saveWindowState()

        # Open the specified file
        try:
            wsf = open(filename, 'wb')
        except: # Report an error message if could not open file
            self.ui.statusBar.showMessage('Erorr: Could not open file', 5000)
            return

        # Write the pickled state dictionary into the file and close it
        pickle.dump(data, wsf)
        wsf.close()

    def getDataState(self):
        dataDict = {}
        dataDict['DATASTRINGS'] = self.DATASTRINGS
        dataDict['ABBRV_DSTR_DICT'] = self.ABBRV_DSTR_DICT
        dataDict['DATADICT'] = self.DATADICT
        dataDict['UNITDICT'] = self.UNITDICT
        dataDict['cstmVars'] = self.cstmVars
        dataDict['colorPlotInfo'] = self.colorPlotInfo
        return dataDict

    def getTimeState(self):
        timeDict = {}
        timeDict['TIMES'] = self.TIMES
        timeDict['TIMEINDEX'] = self.TIMEINDEX
        timeDict['RESOLUTION'] = self.resolution
        timeDict['WIN_TIMES'] = ((self.minTime, self.maxTime), (self.tO, self.tE))
        timeDict['WIN_TICKS'] = ((0, self.iiE), (self.iO, self.iE))

        return timeDict

    def getSelectState(self):
        # Get state information for each object if it is open
        savedRegionState = None
        batchSelectState = None
        if self.fixedSelect is not None:
            savedRegionState = self.fixedSelect.getState()
        if self.batchSelect is not None:
            batchSelectState = self.batchSelect.getState()

        # Store state info in dictionaries
        state = {}
        state['SavedRegion'] = savedRegionState
        state['BatchSelect'] = batchSelectState

        return state

    def getToolState(self):
        toolsInfo = {}

        # Save selection info / state info for tools requiring user time selections
        if self.currSelect and self.currSelect.isFullySelected():
            # Get currently selected tool and regions selected
            name, regions = self.currSelect.getSelectionInfo()
            abbrv = self.toolNameMap[name] # Abbreviated tool name

            # Get the tool associated with this selection and 
            # if applicable, save any state information
            tool = self.tools[abbrv]
            toolState = None
            if tool and hasattr(tool, 'getState'):
                toolState = tool.getState()

            # Store in state dictionary
            toolsInfo['SelectTool'] = (abbrv, regions, toolState)

        # Save information related to stand-alone tools
        genTools = []
        for tool in ['Edit', 'Data', 'PlotMenu']:
            toolObj = self.tools[tool]
            toolState = None
            if toolObj and not toolObj.isHidden():
                genTools.append((tool, toolState))
        toolsInfo['General Tools'] = genTools

        return toolsInfo

    def getFileState(self):
        # Save info about open files, their file type, and if they are ASCII
        # files, save information about the text formatting
        state = {}
        fileType = self.FIDs[-1].getFileType()
        stateInfo = self.FIDs[-1].stateInfo if fileType == 'ASCII' else None 
        state['Names'] = [fd.getName() for fd in self.FIDs]
        state['FileType'] = fileType
        state['Desc'] = stateInfo
        return state

    def getGridOptions(self):
        # Save bool values indicating whether each option in the 'Options' menu
        # is checked
        opts = {}
        opts['Antialias'] = self.ui.antialiasAction.isChecked()
        opts['BridgeGaps'] = self.ui.bridgeDataGaps.isChecked()
        opts['Points'] = self.ui.drawPoints.isChecked()
        opts['AutoScale'] = self.ui.scaleYToCurrentTimeAction.isChecked()

        return opts

    def getEditState(self):
        # Edit names, window edit #, and edit history info
        state = {}
        state['editHistory'] = self.editHistory
        state['currentEdit'] = self.currentEdit
        state['editNames'] = self.editNames

        # If edit window is open, get edit history directly from here since
        # it is not updated until the window is closed
        if self.tools['Edit'] is not None:
            state['editHistory'] = self.tools['Edit'].getEditHistory()

        return state

    def getPlotState(self):
        state = {}
        # Get information about plotted variables, linked plots, heights,
        # and custom pens used
        state['lastPlotStrings'] = self.lastPlotStrings
        state['lastPlotLinks'] = self.lastPlotLinks
        state['lastPlotHeightFactors'] = self.lastPlotHeightFactors
        state['customPens'] = self.mapCustomPens(self.customPens)

        # Check if there are any label sets being displayed and save
        # the list of variables
        pltGrdLblSet = self.pltGrd.labelSetLabel
        lblSet = pltGrdLblSet.dstrs if pltGrdLblSet else []
        state['LabelSets'] = lblSet

        return state

    def saveWindowState(self):
        state = {}
        state['Mode'] = self.insightMode

        # State info keywords and functions to call to get the appropr. state info
        kws = ['Data', 'Time', 'Files', 'Plots', 'Tools', 'Edits',
            'Select', 'GridOptions']
        funcs = [self.getDataState, self.getTimeState, self.getFileState,
            self.getPlotState, self.getToolState, self.getEditState,
            self.getSelectState, self.getGridOptions]

        # Fill dictionary using the keywords and functions above
        for kw, func in zip(kws, funcs):
            state[kw] = func()

        return state

    def mapCustomPens(self, customPens):
        # Maps pens to information about the pen, such as color, style, width, etc.
        # instead of storing the pen directly
        newList = []
        for penList in customPens:
            newPenList = []
            for dstr, en, pen in penList:
                newPenList.append((dstr, en, self.getPenInfo(pen)))
            newList.append(newPenList)

        return newList

    def loadCustomPens(self, customPens):
        # Maps all pen information to pen objects
        newList = []
        for penList in customPens:
            newPenList = []
            for dstr, en, penInfo in penList:
                newPenList.append((dstr, en, self.makePen(*penInfo)))
            newList.append(newPenList)
        return newList

    def loadEditState(self, state):
        # Load in information about previous edits, edit names, & current edit
        for kw in state:
            setattr(self, kw, state[kw])

    def loadPlotState(self, plotState):
        # Set plot info and map custom pens to objects
        self.customPens = self.loadCustomPens(plotState['customPens'])
        for kw in ['lastPlotStrings', 'lastPlotLinks', 'lastPlotHeightFactors']:
            setattr(self, kw, plotState[kw])

        # Plot data using plot info
        pltFactors = self.lastPlotHeightFactors
        self.plotData(self.lastPlotStrings, self.lastPlotLinks, pltFactors)

        # Add in any label sets that were visible
        for label in plotState['LabelSets']:
            self.pltGrd.addLabelSet(label)

    def loadGridOptions(self, opts):
        actions = [self.ui.antialiasAction, self.ui.bridgeDataGaps,
            self.ui.drawPoints, self.ui.scaleYToCurrentTimeAction]
        kws = ['Antialias', 'BridgeGaps', 'Points', 'AutoScale']
        for kw, act in zip(kws, actions):
            act.setChecked(opts[kw])

    def loadSelectState(self, state):
        if state['SavedRegion'] is not None:
            self.openFixedSelection()
            self.fixedSelect.loadState(state['SavedRegion'])
        elif state['BatchSelect'] is not None:
            self.openBatchSelect(closeTools=False)
            self.batchSelect.loadState(state['BatchSelect'])

    def loadToolState(self, toolStateInfo):
        # Load tools that require user selections
        if 'SelectTool' in toolStateInfo:
            # Extract tool state info
            name, regions, toolState = toolStateInfo['SelectTool']

            # Start up tool (initialize object without opening)
            startFunc = self.toolFuncs[name]
            startFunc()

            # Set up selection, loading tool state after minor updates based
            # on time edits but before plots are updated / opened
            self.currSelect.loadToolFromState(regions, self.tools[name], toolState)

        # Load stand-alone tools like Edit, Plot Menu, etc.
        if 'General Tools' in toolStateInfo:
            for toolName, toolState in toolStateInfo['General Tools']:
                startFunc = self.toolFuncs[toolName]
                startFunc()
                if toolState:
                    self.tools[toolName].loadState(toolState)

    def closeAllSubWindows(self):
        self.endLoadDialog()
        for tool in self.toolAbbrv:
            self.closeTool(tool)
        self.closeFixSelection()
        self.closeTimeSelect()
        self.closeBatchSelect()

    def closePlotTools(self):
        for tool in self.select_opts:
            self.closeTool(tool)

    def initVariables(self):
        """init variables here that should be reset when file changes"""
        self.lastPlotStrings = []
        self.lastPlotLinks = None
        self.lastPlotHeightFactors = None
        self.selectMode = None
        self.currentEdit = 0 # current edit number selected
        self.editNames = [] # list of edit names, index into list is edit number
        self.editHistory = []
        self.changeLog = {}
        self.customPens = []
        self.pltGrd = None
        self.regions = []
        self.cstmVars = []

    def updateSelectionMenu(self):
        # Set certain selection tools visible/hidden depending on
        # whether a selection has been made or is to be made
        batchVisible = False
        selectTools = False
        fixSelectVisible = False

        if not self.currSelect: # If no selection, all visible except for saving region
            batchVisible = True
            selectTools = True
            fixSelectVisible = False
        else: # If there is a selection that's been made, hide all except fix select
            batchVisible = False
            selectTools = False
            fixSelectVisible = True

        if self.fixedSelect: # If fixed selection, hide all other tools
            batchVisible = False
            selectTools = False
            fixSelectVisible = True

        if self.batchSelect:
            selectTools = False
            batchVisible = True
            fixSelectVisible = False

        self.ui.actionBatchSelect.setVisible(batchVisible)
        for act in [self.ui.actionSelectByTime, self.ui.actionSelectView]:
            act.setVisible(selectTools)
        self.ui.actionFixSelection.setVisible(fixSelectVisible)

    def closePlotMenu(self):
        if self.tools['PlotMenu']:
            self.tools['PlotMenu'].close()
            self.tools['PlotMenu'] = None

    def closeEdit(self):
        if self.tools['Edit']:
            self.tools['Edit'].close()
            self.tools['Edit'] = None

    def closeData(self):
        if self.tools['Data']:
            self.tools['Data'].close()
            self.tools['Data'] = None

    def closeTraceStats(self):
        if self.tools['Stats']:
            self.tools['Stats'].close()
            self.tools['Stats'] = None

    def closeSpectra(self):
        if self.tools['Spectra']:
            self.tools['Spectra'].close()
            self.tools['Spectra'] = None

    def closeHelp(self):
        if self.helpwin:
            self.helpwin.close()
            self.helpwin = None

    def closeAbout(self):
        if self.aboutDialog:
            self.aboutDialog.close()
            self.aboutDialog = None

    def startPlaneNormal(self):
        self.endGeneralSelect()
        self.closeTool('PlaneNormal')

        # Get color and label
        color = '#039dfc'
        name = 'PlaneNormal'
        label = 'PlaneNormal'
        select_type = 'Single'

        # Create show/close functions
        showFunc = functools.partial(self.showTool, name)
        closeFunc = functools.partial(self.closeTool, name)

        # Create tool
        self.tools['PlaneNormal'] = PlaneNormal(self)

        # Start general selection
        self.initGeneralSelect(label, color, None, 
            select_type, startFunc=showFunc, updtFunc=None, 
            closeFunc=None)
    
    def openMMSOrbit(self):
        self.closeMMSOrbit()
        self.tools['MMSOrbit'] = mms_orbit.MMS_Orbit(self)
        self.tools['MMSOrbit'].show()

    def openMMSFormation(self):
        self.closeMMSFormation()
        self.tools['MMSFormation'] = mms_formation.MMS_Formation(self)
        self.tools['MMSFormation'].show()

    def closeMMSFormation(self):
        if self.tools['MMSFormation']:
            self.tools['MMSFormation'].close()
            self.tools['MMSFormation'] = None

    def closeMMSOrbit(self):
        if self.tools['MMSOrbit']:
            self.tools['MMSOrbit'].close()
            self.tools['MMSOrbit'] = None

    def openMMSData(self):
        self.closeMMSData()
        self.tools['MMSData'] = MMSDataDownloader(self)
        self.tools['MMSData'].show()

    def closeMMSData(self):
        if self.tools['MMSData']:
            self.tools['MMSData'].close()
            self.tools['MMSData'] = None

    def startTool(self, name, trigger=None, initFunc=None):
        # End current selection to start new one
        self.endGeneralSelect()

        # Close tool if previously opened
        self.closeTool(name)

        # Get parameters specific to tool
        ## Selection color and labrl
        color = self.select_colors.get(name)
        label = self.toolAbbrToName[name]
        
        ## Selection type
        if name in self.tool_select_types:
            select_type = self.tool_select_types[name]
        else:
            select_type = 'Single'

        ## Update function
        if name in self.tool_updt_funcs:
            updt_func = self.tool_updt_funcs[name]
        else:
            updt_func = None

        ## Widget class
        WidgetClass = self.widget_classes[name]

        # Create tool
        tool = WidgetClass(self)
        tool.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.tools[name] = tool

        # Create show/close functions
        showFunc = functools.partial(self.showTool, name)
        closeFunc = functools.partial(self.closeTool, name)

        # Apply any other functions before starting select
        if initFunc is not None:
            initFunc(tool)

        # Start general selection
        self.initGeneralSelect(label, color, tool.ui.timeEdit, 
            select_type, startFunc=showFunc, updtFunc=updt_func, 
            closeFunc=closeFunc)
        
        return tool
    
    def setWaveAnalysis(self, plot_type, tool):
        ''' Start wave analysis with plot_type as
            the default plot type if given
        '''
        tool.setPlotType(plot_type)

    def showTool(self, name):
        ''' Displays tool if present '''
        if self.tools[name]:
            self.tools[name].show()
            self.tools[name].update()

    def closeTool(self, name):
        ''' Closes tool if exists '''
        if self.tools[name]:
            self.tools[name].close()
            self.tools[name] = None

    def openPlotMenu(self):
        self.closePlotMenu()
        self.tools['PlotMenu'] = PlotMenu(self)

        geo = self.geometry()
        self.tools['PlotMenu'].move(geo.x() + 200, geo.y() + 100)
        self.tools['PlotMenu'].show()

    def openEdit(self):
        self.closeTraceStats()
        self.closeEdit()
        self.tools['Edit'] = Edit(self)
        self.tools['Edit'].show()

    def fixSelection(self):
        if self.currSelect is None or self.currSelect.regions == []:
            return

        # Initialize interface
        self.closeFixSelection()
        self.openFixedSelection()

        # Adjust the time/lines to match current selection
        strt = self.currSelect.timeEdit.start.dateTime()
        end = self.currSelect.timeEdit.end.dateTime()
        self.fixedSelect.setTimeEdit(strt, end)

    def openFixedSelection(self):
        self.fixedSelect = FixedSelection(self)

        # Create a linked region object
        self.savedRegion = GeneralSelect(self, 'Single', 'Saved Selection', '#595959', 
            self.fixedSelect.ui.timeEdit, None, closeFunc=self.closeFixSelection)
        self.savedRegion.setLabelPos('bottom')
        self.savedRegion.addRegion(0, 0)
        self.fixedSelect.show()

    def closeSavedRegion(self):
        if self.savedRegion:
            self.savedRegion.closeAllRegions(closeTool=False)
            self.savedRegion = None

    def closeFixSelection(self):
        if self.fixedSelect:
            # Clear saved linked region and interface
            self.closeSavedRegion()
            self.fixedSelect.close()
            self.fixedSelect = None
            self.updateSelectionMenu()

    def openTimeSelect(self):
        self.closeTimeSelect()
        self.timeSelect = TimeRegionSelector(self)
        self.timeSelect.show()

    def closeTimeSelect(self):
        if self.timeSelect:
            self.timeSelect.close()
            self.timeSelect = None

    def selectTimeRegion(self, t0, t1):
        if self.currSelect is None:
            self.startTool('Stats')

        if self.currSelect:
            self.currSelect.addRegion(t0-self.tickOffset, t1-self.tickOffset)

    def showData(self):
        # show error message for when loading cdfs because not ported yet
        if not self.FIDs:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Data display not working with CDFs yet")
            msg.setWindowTitle("Error")
            msg.exec_()
            return

        self.closeData() # this isnt actually needed it still closes somehow?
        self.tools['Data'] = DataDisplay(self, self.FIDs, Title='Flatfile Data')
        self.tools['Data'].show()

    def openHelp(self):
        self.closeHelp()
        self.helpwin = HelpWindow(self)
        self.helpwin.show()

    def openAbout(self):
        self.aboutDialog = aboutdialog(NAME, VERSION, COPYRIGHT, self)
        self.aboutDialog.show()

    def toggleAntialiasing(self):
        pg.setConfigOption('antialias', self.ui.antialiasAction.isChecked())
        self.replotData()
        if self.tools['Spectra']:
            self.tools['Spectra'].update()

    def getPrunedData(self, dstr, en, a, b):
        """returns data with error values removed and nothing put in their place (so size reduces)"""
        data = self.getData(dstr, en)[a:b]
        return data[data < self.errorFlag]

    def swapMode(self): #todo: add option to just compile to one version or other with a bool swap as well
        """swaps default settings between marspy and magpy"""
        self.insightMode = not self.insightMode
        self.swapModeUI() # Update user interface and menus based on new mode
        self.plotDataDefault() # Replot data according to defaults for new mode

    def swapModeUI(self):
        # Swap action text
        txt = 'Switch to MMS' if self.insightMode else 'Switch to MarsPy'
        tooltip = 'Loads various presets specific to the MMS mission and better for general use cases' if self.insightMode else 'Loads various presets specific to the Insight mission'
        self.ui.switchMode.setText(txt)
        self.ui.switchMode.setToolTip(tooltip)

        # Hide or show MMS tools menu
        self.ui.showMMSMenu(not self.insightMode)

        # Set window title and icons
        self.setWindowTitle('MarsPy' if self.insightMode else 'MagPy4')
        self.app.setWindowIcon(self.marsIcon if self.insightMode else self.magpyIcon)

    def runTests(self):
        Tests.runTests()

    def saveFileDialog(self, defSfx='.txt', defFilter='TXT file', appendSfx=True):
        defaultSfx = defSfx
        defFilter = defFilter + '(*'+defSfx+')'
        QQ = QtWidgets.QFileDialog(self)
        QQ.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        path = os.path.expanduser(".")
        QQ.setDirectory(path)
        fullname = QQ.getSaveFileName(parent=None, directory=path, caption='Save Data', filter=defFilter)
        if fullname is None:
            print('Save failed')
            return
        if fullname[0] == '':
            print('Save cancelled')
            return

        # If file name doesn't end with default suffix, add it before saving
        filename = fullname[0]
        if filename.endswith(defaultSfx) == False and appendSfx == True:
            filename += defaultSfx

        return filename

    def validFilenames(self, fileNames):
        for fileName in fileNames:
            if '.' not in fileName: # lazy extension check
                print(f'Bad file found, cancelling open operation')
                return False
        if not fileNames:
            print(f'No files selected, cancelling open operation')
            return False

        return True

    def openFileDialog(self, filter_type=None, clear=True):
        # Get flags and dialog class
        flags = QtWidgets.QFileDialog.ReadOnly
        dialog = QtWidgets.QFileDialog

        # Determine label and filter string
        caption_prefix = 'Open' if clear == True else 'Add'
        if filter_type in file_types_dict:
            caption = f'{caption_prefix} {filter_type}'
            filt = file_types_dict[filter_type]
        else:
            caption = f'{caption_prefix} File'
            filt = ';;'.join(sorted(list(file_types_dict.values())))
            filt = 'All Files (*);;' + filt

        # Get selected files
        files = dialog.getOpenFileNames(caption=caption, options=flags,
            filter=filt)[0]

        # Skip if nothing selected
        if len(files) < 0:
            return

        # Separate CDFs from ASCII / Flat Files
        cdfs, others = [], []
        for f in files:
            if f.endswith('.cdf'):
                cdfs.append(f)
            else:
                others.append(f)

        # Try to open list of files
        self.openFileList(others, clear=clear)
        self.addCDF(cdfs, clearPrev=clear)

    def addFileDialog(self, filter_type=None):
        self.openFileDialog(filter_type=filter_type, clear=False)

    def openFlatFile(self):
        self.openFileDialog(filter_type='FLATFILE')

    def openASCII(self):
        self.openFileDialog(filter_type='ASCII')

    def openCDF(self):
        self.openFileDialog(filter_type='CDF')

    def openFileList(self, fileNames, clear):
        # Get basenames for each file path and determine file type
        bases = [os.path.basename(p) for p in fileNames]
        files = {'ff' : [], 'ascii' : [] }
        for name, path in zip(bases, fileNames):
            if '.' not in name or name.endswith('.ffh') or name.endswith('.ffd'):
                files['ff'].append(path)
            else:
                files['ascii'].append(path)

        # Assemble list of files to load and their read functions
        read_funcs = [load_text_file] * len(files['ascii'])
        read_funcs += [load_flat_file] * len(files['ff'])
        load_files = files['ascii'] + files['ff']

        # Skip if nothing to load
        if len(load_files) == 0:
            return

        # Load files asynchronously
        self.loadFilesAsync(read_funcs, load_files, clear)
    
    def loadFilesAsync(self, read_funcs, files_to_read, clear, *args, **kwargs):
        ''' Creates a task to read in the files and connects it to the load dialog '''
        task = TaskThread(read_files, read_funcs, files_to_read, *args ,**kwargs,
            update_progress=True, interrupt_enabled=True)
        self._loadAsync(files_to_read, task, clear)
    
    def loadCDFsAsync(self, clear, *args, **kwargs):
        ''' Creates a task to read in the files and connects it to the load dialog '''
        task = TaskThread(read_cdf_files, *args, **kwargs, update_progress=True,
            interrupt_enabled=True)
        self._loadAsync(args[0], task, clear)
    
    def _loadAsync(self, files_to_read, task, clear):
        ''' Start the task and connect results to load / update functions '''
        load_func = functools.partial(self.loadFileDatas, clear=clear)
        task.signals.result.connect(load_func)
        task.signals.progress.connect(self.updateLoadDialog)
        self.showLoadDialog(files_to_read)
        self.pool.start(task)

    def showLoadDialog(self, file_list=[]):
        ''' Shows a progress dialog for the files being read '''
        self.closeLoadDialog()

        # Create a progress dialog
        self.load_dialog = FileLoadDialog()

        # Create the progress checklist and wrap it in a scroll area
        items = [os.path.basename(n) for n in file_list]
        checklist = ProgressChecklist(items)
        wrapper = ScrollArea()
        wrapper.setWidget(checklist)
        wrapper.setWidgetResizable(True)
        wrapper.adjustToWidget()

        # Show progress checklist in layout if more than one file
        show = len(file_list) > 1
        self.load_dialog.setWidget(wrapper, show=show)

        # Set label
        self.load_dialog.setLabelText('Reading: ')

        # Use a labeled status bar
        bar = LabeledProgress()
        self.load_dialog.setBar(bar)

        # Set attributes
        self.load_dialog.resize(500, 100)
        self.load_dialog.setWindowTitle('Loading Files')
        self.load_dialog.setMinimum(0.0)
        self.load_dialog.setMaximum(0.0)

        # Connect cancel to canceling a file load
        self.load_dialog.canceled.connect(self.cancelFileLoad)

        # Show the dialog
        self.load_dialog.show()

    def updateLoadDialog(self, progress):
        ''' Updates the progress dialog for a file being loaded '''
        if self.load_dialog is None:
            return

        # Unpack progress and update label or scroll to current file
        filename, val, n, code = progress
        if n < 2:
            self.load_dialog.setLabelText(f'Reading {filename}') # Set message
        else:
            self.load_dialog.getWidget().scrollToRow(val)

        # Update progress checklist states
        self.load_dialog.getWidget().widget().setStatus(val, code)

        # Update count ('1 of n') state
        self.load_dialog.getBar().setText(f'{val+1} of {n}')

    def loadFileDatas(self, datas, clear=False):
        ''' Loads a list of (FileData, reader) tuples by calling loadFileData '''
        # Skip if nothing to load
        if len(datas) == 0:
            self.closeLoadDialog()
            return

        # Clear previous files if necessary
        if clear:
            self.epoch = None
            for f in self.FIDs:
                del f
            self.FIDs = []
            self.initDataStorageStructures()

        # Load each data object
        for data in datas:
            self.loadFileData(data)

        # Finish loading and opening files
        self.closeLoadDialog()
        self.finishOpenFileSetup()
    
    def closeLoadDialog(self):
        ''' Close the file loading dialog '''
        if self.load_dialog:
            # Check if any files failed to load and show dialog if so
            # (must not have resulted from a cancel operation)
            if not self.load_dialog.wasCanceled():
                failures = self.load_dialog.getWidget().widget().getFailures()
                if len(failures) > 0:
                    self.notifyOfFailures(failures)

            # Close load dialog and clear
            self.load_dialog.close()
            self.load_dialog = None

    def endLoadDialog(self):
        ''' Close load dialog and signal it to cancel '''
        if self.load_dialog:
            self.load_dialog.cancel()
        self.closeLoadDialog()

    def notifyOfFailures(self, failures):
        ''' Notify the user of any files that failed to load with
            a QDialog box
        '''
        fail_dialog = QtWidgets.QMessageBox(self)
        fail_dialog.setWindowTitle('Error')
        msg = f'The following files could not be read: \n'
        msg += '\n'.join(failures)
        fail_dialog.setText(msg)
        fail_dialog.show()
    
    def cancelFileLoad(self):
        ''' Request to cancel the threads in the pool for loading files '''
        self.pool.terminate()

    def afterFileRead(self):
        # Get field and position groups and add to VECGRPS
        b_grps, pos_grps = find_vec_grps(self.DATASTRINGS)
        b_grps = {f'B{key}':val for key, val in b_grps.items()}
        pos_grps = {f'R{key}':val for key, val in pos_grps.items()}
        self.VECGRPS.update(b_grps)
        self.VECGRPS.update(pos_grps)

    def finishOpenFileSetup(self):
        # Set up view if first set of files has been opened
        if self.startUp:
            self.ui.setupView()
            self.startUp = False
            self.ui.showMMSMenu(not self.insightMode)
            self.setWindowTitle('MarsPy' if self.insightMode else 'MagPy4')
            self.updateSelectionMenu()

        # Update menu items
        self.enableToolsAndOptionsMenus(True)

        # Show export file option only if single file is open
        self.ui.actionExportDataFile.setVisible((len(self.FIDs) == 1))

        # Close sub-windows, re-initialize variables, and replot defaults
        self.closeAllSubWindows()
        self.initVariables()
        self.plotDataDefault()
        self.updateFileLabel()
        self.workspace = None
    
    def updateFileLabel(self):
        files = [os.path.basename(id.name) for id in self.FIDs]
        self.ui.fileStatus.set_labels(files)
    
    def loadFileData(self, read_result):
        if read_result is None:
            return

        file_data, file_id = read_result

        # Set error flag
        flag = file_data.get_flag()
        if self.errorFlag is not None:
            self.errorFlag = min(flag, self.errorFlag)
        else:
            self.errorFlag = flag

        # Get labels, units, resolution
        res = file_data.get_resolution()
        units = file_data.get_num_units()
        times = file_data.get_times()
        labels = file_data.get_numerical_keys()

        # Set epoch and map times if previous epoch does not match
        epoch = file_data.get_epoch()
        if self.epoch is not None and epoch != self.epoch:
            dates = ff_time.ticks_to_dates(times, epoch)[0]
            times = ff_time.dates_to_ticks(dates, self.epoch, fold_mode=True)
        else:
            self.epoch = epoch

        # Set resolution
        self.resolution = min(self.resolution, res)

        # Update vector groups
        self.VECGRPS.update(file_data.vec_grps)

        # Assemble data into list format
        datas = file_data.get_numerical_data()
        specs = file_data.get_spec_datas()
        self.loadData(times, labels, datas, units, specDatas=specs)

        # Add ID to list of open files
        if file_id not in self.FIDs:
            self.FIDs.append(file_id)

        # Finish file opening
        self.afterFileRead()
        self.updateAfterOpening()

    def openTextFile(self, filename):
        ''' Open an ASCII file '''
        data = load_text_file(filename, self.epoch)
        self.loadFileData(data)

    def updateAfterOpening(self):
        self.calculateAbbreviatedDstrs()
        self.calculateTimeVariables()

    def rmvCustomVar(self, dstr):
        if dstr not in self.DATASTRINGS or dstr not in self.ABBRV_DSTR_DICT:
            return

        # Remove from datatstrings list
        self.DATASTRINGS.remove(dstr)
        self.ABBRV_DSTR_DICT.pop(dstr)

        # Remove time index; TODO: How to remove TIMES w/o affecting other indices
        ti = self.TIMEINDEX[dstr]
        self.TIMEINDEX.pop(dstr)

        # Remove data from dictionaries
        for d in [self.ORIGDATADICT, self.DATADICT, self.UNITDICT]:
            d.pop(dstr)

    def initNewVar(self, dstr, dta, units='', times=None):
        # Add new variable name to list of datastrings
        if dstr not in self.cstmVars:
            self.cstmVars.append(dstr)
            self.DATASTRINGS.append(dstr)
        self.ABBRV_DSTR_DICT[dstr] = dstr

        # Use any datastring's times as base
        if times is None:
            times = self.getTimes(self.DATASTRINGS[0], 0)
        

        # Mask out any errors in calculated data and apply mask to times
        t, diff, res = times
        mask1 = ~np.isnan(dta)
        mask2 = np.abs(dta) < self.errorFlag
        mask = np.logical_and(mask1, mask2)
        t = t[mask] if len(t) == len(dta) else t
        diff = np.diff(t, append=[diff[0]])
        dta = dta[mask]

        # Clip data if there is a difference
        n = min(len(t), len(dta))
        t = t[:n]
        diff = diff[:n]
        dta = dta[:n]
        times = (t, diff, res)

        # Add in data to dictionaries, no units
        self.TIMES.append(times)
        self.TIMEINDEX[dstr] = len(self.TIMES) - 1
        self.ORIGDATADICT[dstr] = dta
        self.DATADICT[dstr] = [dta]
        self.UNITDICT[dstr] = units

        # Pad rest of datadict to have same length
        length = len(self.editHistory)
        while len(self.DATADICT[dstr]) <= length:
            self.DATADICT[dstr].append([])

    def initDataStorageStructures(self):
        """
        initializes all data structures used for storing loaded data
        file opening operations can either append to or re init these structures based on open preference
        """

        self.DATASTRINGS = [] # list of all the original data column string names of the loaded files (commonly referenced as 'dstrs')
        self.ABBRV_DSTR_DICT = {} # dict mapping dstrs to simplified abbreviated dstr that get generated for easier display
        self.ORIGDATADICT = {} # dict mapping dstrs to original data array
        self.DATADICT = {}  # dict mapping dstrs to lists of data where each element is a list of edited data (the first being unedited)
        self.UNITDICT = {} # dict mapping dstrs to unit strings
        self.SPECDICT = {}
        self.CDFSPECS = {}
        self.TIMES = [] # list of time informations (3 part lists) [time series, resolutions, average res]
        self.TIMEINDEX = {} # dict mapping dstrs to index into times list
        self.EDITEDTIMES = {}
        self.VECGRPS = {}

        self.minTime = None # minimum time tick out of all loaded times
        self.maxTime = None # maximum
        self.iiE = None # maximum tick of sliders (min is always 0)
        self.resolution = 1000000.0 # this is set to minumum resolution when files are loaded so just start off as something large
        self.colorPlotInfo = {}

    def loadData(self, ffTime, newDataStrings, datas, units, specDatas={}):
        # if all data strings currently exist in main data then append everything instead
        allMatch = True
        for dstr in newDataStrings:
            if dstr not in self.DATADICT:
                allMatch = False
                break

        if allMatch: # not sure if all dstrs need to necessarily match but simplest for now
            ncols = len(newDataStrings)
            timeIndices = []
            genDstr = newDataStrings[0]
            old_times = self.TIMES[self.TIMEINDEX[genDstr]][0]
            for i in range(0, ncols):
                dstr = newDataStrings[i]
                data = datas[i]
                old_data = self.ORIGDATADICT[dstr]
                merged_times, merged_data = merge_datas(old_times, ffTime, old_data, data, self.errorFlag)
                self.ORIGDATADICT[dstr] = merged_data
                self.DATADICT[dstr] = [Mth.interpolateErrors(self.ORIGDATADICT[dstr],self.errorFlag)]

                if (i == 0) and (not np.array_equal(merged_times, old_times)):
                    res, diffs = get_resolution_diffs(merged_times)
                    diffs = np.concatenate([diffs[1:], [diffs[-1]]])
                    self.TIMES[self.TIMEINDEX[dstr]] = (merged_times, diffs, res)

            # Merge specDatas passed
            for spec in specDatas:
                # Add to dictionary if not present
                if spec not in self.CDFSPECS:
                    self.CDFSPECS[spec] = specDatas[spec]
                    continue

                grid = specDatas[spec].get_grid()
                y_bins, x_bins = specDatas[spec].get_bins(padded=False)

                # Get old spec bins and grid info
                oldSpec = self.CDFSPECS[spec]
                old_y, old_x = oldSpec.get_bins(padded=False)
                old_grid = oldSpec.get_grid()

                # If 2D y bins
                if not oldSpec.single_y_bins():
                    # Merge y bins as well
                    data = np.hstack([y_bins, (grid.T)])
                    oldData = np.hstack([old_y, old_grid.T])
                    merged_x, merged_data = merge_datas(old_x, x_bins, oldData, data, self.errorFlag)
                    new_y = merged_data[:,0:len(old_y[0])]
                    new_grid = merged_data[:,len(old_y[0]):].T
                    oldSpec.set_data(new_y, new_grid, merged_x)
                else:
                    # Otherwise just merge grid
                    data = grid.T
                    oldData = old_grid.T
                    merged_x, merged_data = merge_datas(old_x, x_bins, oldData, data, self.errorFlag)
                    oldSpec.set_data(y_bins, merged_data.T, merged_x)

        else: # this is just standard new flatfile, cant be concatenated with current because it doesn't have matching column names
            self.DATASTRINGS.extend(newDataStrings)
            avgRes, resolutions = get_resolution_diffs(ffTime)
            resolutions = np.append(resolutions[1:], resolutions[-1]) # append last value to make same length as time series
            self.resolution = min(self.resolution, avgRes)
            self.TIMES.append([ffTime, resolutions, avgRes])

            for i, dstr in enumerate(newDataStrings):
                self.TIMEINDEX[dstr] = len(self.TIMES) - 1 # index is of the time series we just added to end of list
                self.ORIGDATADICT[dstr] = datas[i]
                self.DATADICT[dstr] = [Mth.interpolateErrors(self.ORIGDATADICT[dstr],self.errorFlag)]
                self.UNITDICT[dstr] = units[i]

            self.CDFSPECS.update(specDatas)

        return True

    # redos all the interpolated errors, for when the flag is changed
    def reloadDataInterpolated(self):
        for dstr in self.DATASTRINGS:
            self.DATADICT[dstr] = [Mth.interpolateErrors(self.ORIGDATADICT[dstr],self.errorFlag)]

    def calculateTimeVariables(self):
        self.minTime = None # temp reset until figure out better way to specify if they want file to be loaded fresh or appended to current loading
        self.maxTime = None
        # iterate over times and find min and max???
        for times,_,_ in self.TIMES:
            assert(len(times) > 2)
            self.minTime = times[0] if not self.minTime else min(self.minTime, times[0])
            self.maxTime = times[-1] if not self.maxTime else max(self.maxTime, times[-1])

        # prob dont edit these two on subsequent file loads..?
        self.iO = 0
        self.iE = int((self.maxTime - self.minTime) / self.resolution)

        tick = 1.0 / self.resolution
        print(f'tick resolution : {self.resolution}')
        print(f'time resolution : {tick} Hz')

        self.tO = self.minTime # currently selected time range
        self.tE = self.maxTime

        print(f'tO: {self.tO}')
        print(f'tE: {self.tE}')

        self.iiE = self.iE
        print(f'slider ticks: {self.iiE}')

        self.ui.setupSliders(tick, self.iiE, self.getMinAndMaxDateTime())

    def getCDFPaths(self):
        fileNames = QtWidgets.QFileDialog.getOpenFileNames(self, caption="Open CDF File", options = QtWidgets.QFileDialog.ReadOnly, filter='CDF Files (*.cdf)')[0]
        return fileNames

    def addCDF(self, files=None, exclude_keys=[], label_funcs=None,
                clearPrev=False, clip_range=None):
        # Prompt for filenames if none are given
        if files is None:
            files = self.getCDFPaths()

        # Ignore if no files selected
        if len(files) == 0:
            return

        # Load the files asynchronously
        self.loadCDFsAsync(clearPrev, files, exclude_keys=exclude_keys, 
            clip_range=clip_range, label_funcs=label_funcs)

    def calculateAbbreviatedDstrs(self):
        """
        split on '_' and calculate common tokens that appear in each one. then remove and reconstruct remainders
        these abbreviations are used mainly for the cdf strings since those are longer so plots and things wont have super long strings in them
        """
        self.ABBRV_DSTR_DICT = {}

        # common should be union of the splits
        common = None
        for dstr in self.DATASTRINGS:
            splits = dstr.split('_')
            if common:
                common = common & set(splits)
            else:
                common = set(splits)

        #print('\n'.join([n for n in common]))
        for dstr in self.DATASTRINGS:
            splits = dstr.split('_')
            if len(splits) <= 2:
                abbrvStr = dstr
            else:
                abb = []
                for s in splits:
                    if s not in common:
                        abb.append(s)
                abbrvStr = '_'.join(abb)

            self.ABBRV_DSTR_DICT[dstr] = abbrvStr

    def getMinAndMaxDateTime(self):
        minDt = ff_time.tick_to_date(self.minTime, self.epoch)
        maxDt = ff_time.tick_to_date(self.maxTime, self.epoch)
        return minDt, maxDt

    def getCurrentDateTime(self):
        return self.ui.timeEdit.start.dateTime(), self.ui.timeEdit.end.dateTime()
    
    def sliderReleased(self):
        self.tracker.setAllRegionsVisible(False)

    def onStartSliderChanged(self, val):
        self.onSliderChanged(self.ui.timeEdit.start, val)

    def onEndSliderChanged(self, val):
        self.onSliderChanged(self.ui.timeEdit.end, val)
    
    def onSliderChanged(self, edit, val):
        ''' 
            Updates other elements and tracker lines when slider
            buttons dragged
        '''
        # Show tracker line
        if self.tracker:
            self.tracker.setAllRegionsVisible(True)
            self.tracker.setRegion((val+self.minTime, self.maxTime + 1))

        # Update time edit
        d = ff_time.tick_to_date(val+self.minTime, self.epoch)
        edit.blockSignals(True)
        edit.setDateTime(d)
        edit.blockSignals(False)

    def getTimeFromTick(self, tick):
        assert(tick >= 0 and tick <= self.iiE)
        return self.minTime + (self.maxTime - self.minTime) * tick / self.iiE

    # try to find good default plot strings
    def getDefaultPlotInfo(self):
        dstrs = []
        links = []

        # Get filetype mode
        mode = self.FIDs[0].getFileType()

        # If CDF is open and in MMS mode
        if mode == 'CDF' and not self.insightMode:
            # Get MMS variable groups in spacecraft groupings
            # and map to axis grouping and plotStrings format
            grps, btots = get_mms_grps(self)
            grps = grps['Field']
            mms_dstrs = [[],[],[], []]
            for sc_id in [1,2,3,4]:
                if sc_id not in grps:
                    continue
                sc_grp = grps[sc_id]
                for i in range(0, len(sc_grp)):
                    mms_dstrs[i].append((sc_grp[i], 0))
            
            mms_dstrs = [row for row in mms_dstrs if len(row) > 0]
            
            # Check if full set of MMS variables loaded
            if set(list(map(len, mms_dstrs))) == set([4]):
                dstrs = mms_dstrs
                links = [0,1,2,3]

        # If VECGRPS is not empty, try using the first vecgrp
        if len(dstrs) == 0 and len(self.VECGRPS) > 0:
            key = list(self.VECGRPS.keys())[0]
            variables = self.VECGRPS[key]
            dstrs = [[(dstr, 0)] for dstr in variables]
            links = [i for i in range(len(dstrs))]
        
        # MMS trace pen presets
        lstLens = list(map(len, dstrs))
        if not self.insightMode and lstLens != [] and min(lstLens) == 4 and max(lstLens) == 4:
            for dstrLst in dstrs:
                penLst = []
                for (currDstr, en), color in zip(dstrLst, self.mmsColors):
                    penLst.append((currDstr, en, pg.mkPen(color)))
                self.customPens.append(penLst)
        else: # Reset custom pens in case previously set
            self.customPens = []

        # If num plots <= 1 (non-standard file), try to plot first 3 variables
        if not links or len(dstrs) <= 1:
            dstrs = [[(dstr, 0)] for dstr in self.DATASTRINGS[0:3]]
            links = [[i] for i in range(0, len(dstrs))]
        else:
            links = [links]

        return dstrs, links

    def plotDataDefault(self):
        dstrs, links = self.getDefaultPlotInfo()
        heights = []

        # If there is a saved plot default, make sure all 
        if self.savedPlotInfo is not None:
            missing = False
            for plot in self.savedPlotInfo[0]:
                for dstr, en in plot:
                    # Check if special plot
                    if en < 0 and (dstr not in self.SPECDICT):
                        missing = True
                        break
                    # Check if variable is loaded
                    elif en >= 0 and (dstr not in self.DATASTRINGS):
                        missing = True
                        break

            # If all plot variables are loaded, use the saved plot settings
            if not missing:
                dstrs, links, heights = self.savedPlotInfo
            else:
                # Otherwise, reset the saved plot info
                self.savedPlotInfo = None

        numPts = self.plotData(dstrs, links, heights)

        # If a large number of points are plotted, enable downsampling for the plots
        if numPts > 5000:
            self.ui.downsampleAction.setChecked(True)
            msg = "Plot data downsampled; disable under 'Options' Menu"
            self.ui.statusBar.showMessage(msg, 10000)
        else: # Otherwise, disable downsampling
            self.ui.downsampleAction.setChecked(False)

        # Set initial y range if scaling disabled
        if not self.ui.scaleYToCurrentTimeAction.isChecked():
            self.updateYRange(force=True)

    def getData(self, dstr, editNumber=None):
        edits = self.DATADICT[dstr]

        if (editNumber is None or len(edits) <= editNumber):
            i = len(edits) - 1 
        else:
            i = editNumber
        while len(edits[i]) == 0: # if empty list go back one
            i -= 1
        return edits[i]

    def getLabel(self, dstr, editNumber=None):
        edits = self.DATADICT[dstr]
        i = self.currentEdit if editNumber is None else editNumber

        if len(edits) <= i:
            i = len(edits) - 1 
        else:
            i = editNumber

        while len(edits[i]) == 0: # if empty list go back one
            i -= 1
        return dstr if i == 0 else f'{dstr} {self.editNames[i][:8]}'

    def getFileNameString(self, maxLabelWidth): # returns list of all loaded files
        name = 'unknown'
        if len(self.FIDs) > 1:
            names = [os.path.split(FID.name)[1] for FID in self.FIDs]
            firstFile = names[0]
            lastFile = names[-1]
            middleFiles = ', '
            labelLen = len(firstFile) + len(lastFile)

            # For any files in between first and last files
            for currFile in names[1:-1]:
                # If the given file makes the label too long, use an ellipsis and
                # stop iterating through the rest of the filenames
                if (labelLen + len(currFile)) > maxLabelWidth:
                    middleFiles += '. . . , '
                    break
                else:
                # Otherwise, add it to the list and update the label length
                    middleFiles += currFile + ', '
                    labelLen += len(currFile)

            # Return the first/last files and any files in between that fit
            name = firstFile + middleFiles + lastFile
        elif len(self.FIDs) > 0:
            name = self.FIDs[0].name
        return name

    def getAbbrvDstr(self, dstr):
        return self.ABBRV_DSTR_DICT[dstr]

    def onPlotRemoved(self, oldPlt):
        # Remove all linked region items from this plot and GeneralSelect
        # lists before removing the old plot
        for select in [self.currSelect, self.batchSelect, self.savedRegion]:
            if select is not None:
                select.onPlotRemoved(oldPlt)

    def plotData(self, dataStrings, links, heightFactors, save=False):
        # Remove any saved linked regions from plots and save their state
        self.ui.scrollSelect.set_range(self.maxTime-self.minTime)
        self.ui.scrollSelect.set_start(self.tO-self.minTime)
        self.ui.scrollSelect.set_range(self.tE-self.minTime)
        selectState = self.getSelectState()
        self.closeFixSelection()
        self.closeBatchSelect()

        # Clear any selected tools
        self.closeTraceStats()
        self.endGeneralSelect()

        # save what the last plotted strings and links are for other modules
        self.lastPlotStrings = dataStrings
        self.lastPlotLinks = links
        self.lastPlotHeightFactors = heightFactors

        # Save plot strings when loading defaults
        if save:
            self.savedPlotInfo = (dataStrings, links, heightFactors)

        self.plotItems = []
        self.labelItems = []

        # A list of pens for each trace (saved for consistency with spectra)
        self.plotTracePens = []

        # Store any previous label sets (for current file)
        label_sets = []
        if self.pltGrd is not None:
            label_sets = self.pltGrdObject.list_axis_grids()

        # Clear previous grid
        self.ui.glw.clear()

        # Add label for file name at top right
        fileList = [os.path.basename(FID.name) for FID in self.FIDs]
        fileNameLabel = FileLabel(fileList)

        # Create new plot grid
        grid_object = PlotGridObject(self)
        self.pltGrd = grid_object.get_layout()
        self.pltGrdObject = grid_object
        self.ui.glw.addItem(grid_object, 0, 0, 1, 1)

        # Connect plot to time range UI
        grid_object.sigXRangeChanged.connect(self._xrange_updated)

        # Set plot grid attributes
        self.pltGrd.set_height_factors(heightFactors)
        self.pltGrdObject.set_tick_text_size(11)
        self.pltGrdObject.set_plot_styles(tickLength=-10)

        nplots = len(dataStrings)
        npoints = 0
        plot_colors = plot_helper.get_colors(dataStrings)
        self.plotTracePens = plot_colors
        spec_objects_full = []
        for i in range(nplots):
            dstrs = dataStrings[i]
            colors = plot_colors[i]

            # Set up plot
            self.pltGrd.add_row()
            pi = MagPyPlotItem(epoch=self.epoch)
            pi.getAxis('right').setStyle(showValues=False)

            # Set plot styles
            vb = pi.getViewBox()
            vb.set_window(self)
            vb.enableAutoRange(x=False, y=False) # range is being set manually in both directions
            vb.setMouseEnabled(y=False)

            # Plot lines
            points, spec_objects = self.plot_traces(pi, dstrs, colors)
            npoints += points

            # Add horizontal zero line
            zero_line = pg.InfiniteLine(movable=False, angle=0, pos=0)
            zero_line.setPen(pg.mkPen('#000000', width=1, style=QtCore.Qt.DotLine))
            pi.addItem(zero_line, ignoreBounds=True)
            pi.ctrl.logYCheck.toggled.connect(functools.partial(self.updateLogScaling, i))

            # Set up label
            units = [self.UNITDICT[dstr] for dstr, en in dstrs if dstr in self.UNITDICT]
            label = StackedLabel([], [], units, size=12)
            self.pltGrd[i] = [label, pi]

            # Store additional spectrogram graphics items
            if spec_objects is not None:
                spec_objects_full.append((i, spec_objects))
        self.update_plot_labels()
        
        # Add in any spectrogram-related additional objects
        if len(spec_objects_full) > 0:
            self.pltGrd.add_col()
            self.pltGrd.add_col()
            for i, objects in spec_objects_full:
                self.pltGrd[i] = self.pltGrd[i][:2] + [objects[1], objects[2]]

        # Add in previously seen axis grids
        for key in label_sets:
            for label in label_sets[key]:
                x = self.getTimes(label, self.currentEdit)[0]
                y = self.getData(label, self.currentEdit)
                interp = interpolate.interp1d(x, y, bounds_error=False, 
                    fill_value=self.errorFlag)
                self.pltGrdObject.get_layout().add_axis(label, interp, loc=key)

        self.pltGrd.set_x_range(self.tO, self.tE)
        self.pltGrd.set_x_lim(self.minTime, self.maxTime)
        self.pltGrdObject.set_links(links)
        self.pltGrdObject.sigPlotColorsChanged.connect(self.update_plot_colors)

        # Add trackers
        plots = self.pltGrd.get_plots()
        self.tracker = TrackerRegion(self, plots)

        # # Downsample data if checked
        self.enableDownsampling(self.ui.downsampleAction.isChecked())

        return npoints

    def update_plot_colors(self, info):
        # Find plot index trace is in
        changed_plot, label, (old_color, new_color) = info
        plots = self.pltGrdObject.get_plots()
        plot_index = None
        i = 0
        for plot in plots:
            if plot == changed_plot:
                plot_index = i
                break
            i += 1

        if plot_index is None:
            return
        
        # Find corresponding label
        i = 0
        for dstr, en in self.lastPlotStrings[plot_index]:
            if self.getLabel(dstr, en) == label:
                self.plotTracePens[plot_index][i] = new_color.name()
                break
            i += 1

    def split_labels(self, dstrs, colors):
        # Split any spectrogram plot labels onto multiple lines
        # if they go over character limit
        labels = []
        label_colors = []
        for (dstr, en), color in zip(dstrs, colors):
            if en < 0:
                # Split label by spaces
                max_chars = 10
                split_label = dstr.split(' ')

                # Merge lines if less than max chars per line
                row = []
                split_keys = []
                for item in split_label:
                    row.append(item)
                    row_sum = sum(list(map(len, row)))
                    if row_sum > max_chars:
                        split_keys.append(' '.join(row))
                        row = []
                if len(row) > 0:
                    split_keys.append(' '.join(row))

                labels.extend(split_keys)
                label_colors.extend(['#000000']*len(split_keys))
            else:
                labels.append(dstr)
                label_colors.append(color)
        return labels, label_colors

    def plot_traces(self, plot, dstrs, colors):
        points = 0
        bridge_gaps = self.ui.bridgeDataGaps.isChecked()
        scatter = self.ui.drawPoints.isChecked()
        result = None
        for color, (dstr, en) in zip(colors, dstrs):
            # Plot spectrograms separately
            if en < 0:
                result = self.plot_spec_data(plot, dstr)
                continue

            # Get data for trace and label
            data = self.getData(dstr, en)
            time, diffs, res = self.getTimes(dstr, en)
            pen = pg.mkPen(color)
            name = self.getLabel(dstr, en)

            # Scatter plot
            if scatter:
                color = pen.color()
                outline_color = pg.mkPen(color.darker(150))
                brush = pg.mkBrush(color)
                pdi = plot.plot(time, data, pen=None, symbolPen=outline_color,
                    symbolBrush=brush, name=name, symbol='o', symbolSize=6, 
                    pxMode=True)
            # Trace plot
            else:
                if bridge_gaps:
                    segments = 'all'
                else:
                    segments = plot_helper.get_segments(diffs, res)
                plot.plot(time, data, pen=pen, connect=segments, name=name)
            
            points += len(data)
        return points, result
    
    def plot_spec_data(self, plot, key):
        # Find spec data corresponding to key
        spec = None
        if key in self.CDFSPECS:
            spec = self.CDFSPECS[key]
        elif key in self.SPECDICT:
            spec = self.SPECDICT[key]

        # Load spec data
        if spec is not None:
            return plot.load_color_plot(spec)
        return None
    
    def add_spectrogram(self, spec_data, key):
        # Add spec to data dict, add to plot list, then replot
        self.SPECDICT[key] = spec_data
        self.lastPlotStrings.append([(key, -1)])
        self.plotTracePens.append([])
        self.replotGrid()

    def plot_added(self):
        # Updates to grid after plot added
        self.pltGrd.set_x_range(self.tO, self.tE)
        self.pltGrd.set_x_lim(self.minTime, self.maxTime)
        self.pltGrdObject.set_tick_text_size(11)
        self.pltGrdObject.set_plot_styles(tickLength=-10)
        self.pltGrdObject.update_y_ranges()
        if self.tracker:
            self.tracker.add_plot(self.pltGrd.get_plots()[-1])

    def replotData(self):
        plots = self.pltGrd.get_plots()
        for i in range(len(plots)):
            plot = plots[i]
            plot.clear_data()
            dstrs = self.lastPlotStrings[i]
            colors = self.plotTracePens[i]
            self.plot_traces(plot, dstrs, colors)
        self.pltGrdObject.update_y_ranges()
    
    def replotGrid(self):
        self.plotData(self.lastPlotStrings, self.lastPlotLinks, self.lastPlotHeightFactors)

    def update_current_edit(self, old_edit, editnum):
        newPlotStrings = []
        for row in self.lastPlotStrings:
            rowStrings = []
            for dstr, en in row:
                if en < 0:
                    rowStrings.append((dstr, en))
                else:
                    if old_edit >= en:
                        rowStrings.append((dstr, editnum))
                    else:
                        rowStrings.append((dstr, en))
            newPlotStrings.append(rowStrings)
        self.lastPlotStrings = newPlotStrings
        self.replotData()
        self.update_plot_labels()
    
    def update_plot_labels(self):
        labels = self.pltGrd[:,0]

        for row, label, pens in zip(self.lastPlotStrings, labels, self.plotTracePens):
            dstrs = []
            label_colors = []
            for (dstr, en), color in zip(row, pens):
                if en < 0:
                    subdstrs, subcolors = self.split_labels([(dstr, en)], [color])
                    dstrs += subdstrs
                    label_colors += subcolors
                else:
                    dstrs.append(self.getLabel(dstr, en))
                    label_colors.append(color)
            if label is not None:
                label.set_labels(dstrs)
                label.set_colors(label_colors)

    def enableScrolling(self, val):
        # Minimum plot height set to 3 inches for now
        min_height = self.minimumPlotHeight * QtWidgets.QDesktopWidget().logicalDpiY()

        # Set minimum height for gview accordingly
        if val:
            self.ui.gview.setMinimumHeight(min_height * len(self.plotItems) + 100)
        else:
            self.ui.gview.setMinimumHeight(0)

    def increasePlotHeight(self):
        # Increases the minimum plot height used to set gview size
        # when scrolling is enabled
        if self.ui.enableScrollingAction.isChecked():
            self.minimumPlotHeight = min(self.minimumPlotHeight + 0.5, 5)
            self.enableScrolling(True)

    def decreasePlotHeight(self):
        # Calculate the approximate height of window in inches so that
        # the lower bound for the minimum plot height is set correctly
        # otherwise, increasePlotHeight doesn't always affect the view
        winHeight = self.ui.scrollFrame.size().height()
        viewHeightInches = (winHeight / QtWidgets.QDesktopWidget().logicalDpiY()) - 2
        minBound = max(viewHeightInches/len(self.plotItems), 1)

        # Decreases the minimum plot height used to set gview size
        # when scrolling is enabled
        if self.ui.enableScrollingAction.isChecked():
            self.minimumPlotHeight = max(self.minimumPlotHeight - 0.5, minBound)
            self.enableScrolling(True)

    def enableDownsampling(self, val):
        plots = self.pltGrd.get_plots()
        if val:
            for plt in plots:
                plt.setDownsampling(ds=None, auto=True, mode='peak')
                plt.setClipToView(True)
        else:
            for plt in plots:
                plt.setDownsampling(ds=False)
                plt.setClipToView(False)

    def replotDataCallback(self):
        # done this way to ignore the additional information ui callbacks will provide
        self.replotData()

    def getTimes(self, dstr, editNumber):
        times, resolutions, avgRes = self.TIMES[self.TIMEINDEX[dstr]]

        # Check if data has been edited
        if dstr in self.EDITEDTIMES:
            timeDict = self.EDITEDTIMES[dstr]
            editList = sorted(list(timeDict.keys()))

            # Return times for edit >= editNumber
            for en in editList[::-1]:
                if en <= editNumber:
                    return self.EDITEDTIMES[dstr][en]

        # check if arrays arent same length then assume the difference is from a filter operation
        Y = self.getData(dstr, editNumber)
        if len(Y) < len(times):
            diff = len(times) - len(Y) + 1
            times = times[diff // 2:-diff // 2 + 1]
            assert len(Y) == len(times), 'filter time correction failed...'
            resolutions = np.diff(times)
        return times,resolutions,avgRes
    
    def getTimeIndex(self, dstr, en):
        return self.TIMEINDEX[dstr]

    def getViewRange(self):
        return self.tO, self.tE

    def updateLogScaling(self, plotIndex, val):
        # Look through all link sets the plot at the given index is in
        for row in self.lastPlotLinks:
            if plotIndex not in row:
                continue

            # Check link set to see if it contains a log-scaled plot or not
            logModeDetected = False
            for plotIndex in row:
                if self.plotItems[plotIndex].ctrl.logYCheck.isChecked() == val:
                    logModeDetected = True

            # Adjust all other plots to have same scale
            if logModeDetected:
                for plotIndex in row:
                    currPlt = self.plotItems[plotIndex]
                    currPlt.ctrl.logYCheck.blockSignals(True)
                    currPlt.ctrl.logYCheck.setChecked(val)
                    currPlt.updateLogMode()
                    currPlt.ctrl.logYCheck.blockSignals(False)

        # Manually update y ranges
        self.updateYRange()

    def updateYRange(self):
        self.pltGrdObject.set_autoscale(self.ui.scaleYToCurrentTimeAction.isChecked())

    def flagData(self, flag=None):
        if len(self.FIDs) > 1:
            return

        # Get number of selected regions
        numRegions = len(self.currSelect.regions)

        # Stores new times
        timeDict = {}

        en = self.currentEdit
        lastTimes = None
        for dstr in self.DATADICT:
            # Get data and times
            data = np.array(self.getData(dstr, en))
            times = np.array(self.getTimes(dstr, en)[0])

            # Get start/end indices for each region
            regionTicks = []
            for regionNum in range(0, numRegions):
                t0, t1 = self.getSelectionStartEndTimes(regionNum)
                sI = bisect.bisect_right(times, t0)
                eI = bisect.bisect_right(times, t1)
                regionTicks.append((t0, t1))

                # Remove flagged data or replace w/ flag
                if flag is None:
                    if eI >= len(times):
                        times = times[:sI]
                        data = data[:sI]
                    else:
                        times = np.concatenate([times[:sI], times[eI+1:]])
                        data = np.concatenate([data[:sI], data[eI+1:]])
                else:
                    # Fill indices with flag
                    data[sI:eI+1] = flag

            # Update stored data
            self.DATADICT[dstr].append(data)

            # Reuse previous array if same as last
            if lastTimes is not None and np.array_equal(lastTimes, times):
                times = lastTimes

            # Get new time ticks and add to EditedTimes dict if deleted
            if flag is not None:
                continue
            avgRes, resolutions = get_resolution_diffs(times)
            timeInfo = (times, resolutions, avgRes)
            if dstr not in self.EDITEDTIMES:
                self.EDITEDTIMES[dstr] = {en+1:timeInfo}
            else:
                self.EDITEDTIMES[dstr][en+1] = timeInfo

            lastTimes = times

        return regionTicks

    def exportFile(self):
        ''' Opens up file dialog for exporting an edited file '''
        if len(self.FIDs) > 1:
            return

        # Get information about file type
        FID = self.FIDs[0]
        fileType = FID.getFileType()

        # Determine file dialog filter and file generating function based on file type
        fdFilter = ['Flat Files (*.ffh)'] if fileType != 'ASCII' else ['All files (*)']
        exportFunc = self.exportASCII if fileType == 'ASCII' else self.exportFlatFileCopy

        # Generate a default new filename
        if '.' in FID.name:
            basename, extension = FID.name.split('.')
            defaultName = f'{basename}_new.{extension}'
        else:
            defaultName = FID.name + '_new'

        # Get filename and connect file dialog accept to export function
        fd = QtWidgets.QFileDialog(self)
        fd.selectFile(defaultName)
        fd.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        fd.setFileMode(QtWidgets.QFileDialog.AnyFile)
        fd.setNameFilters(fdFilter)
        fd.fileSelected.connect(exportFunc)
        fd.open()

    def exportASCII(self, name):
        ''' Exports latest edited data as an ASCII file '''
        # Get latest records
        records, labels = self.getLatestRecords()
        times = records[:,0]
        data = records[:,1:]

        # Get latest file and reader
        FID = self.FIDs[0]
        reader = FID.get_reader()
        header = reader.get_header().strip('\n')

        # Map ticks to strings again
        date_to_ts = lambda dt : datetime.isoformat(dt)[:-3]
        dates = list(map(self.datetime_from_tick, times))
        ts = list(map(date_to_ts, dates))

        # Determine delimeter and data format based on file type
        fmt = ['%s'] + ['%.7f'] * len(labels)
        if reader.subtype() == 'CSV':
            delim = ','
        else:
            delim = ''
            # Adjust formats based on fixed-width column lengths
            slices = reader.get_slices()

            ## Format first record of data according to format strings
            test_record = [ts[0]] + data[0].tolist()
            test_record = [fmtstr % val for fmtstr, val in zip(fmt, test_record)]

            ## Adjust widths so data will fit in each column
            widths = []
            for test_str, (a, b) in zip(test_record, slices):
                width = max(len(test_str) + 1, b-a)
                widths.append(width)

            # Left-align format strings
            fmt = [f'%-{n}'+fmtstr.split('%')[1] for (n, fmtstr) in zip(widths, fmt)]

            # Split and rejoin header items based on slices
            cols = [header[slice(*s)].strip(' ') for s in slices]
            hdr_fmt = ''.join([f'%-{n}s' for n in widths])
            header = hdr_fmt % tuple(cols)

        # Create a structured array from the data and mapped times
        dtype = [('time', 'U72')] + [(label, 'f8') for label in labels]
        dtype = np.dtype(dtype)
        records = np.hstack([np.vstack(ts), data])
        table = rfn.unstructured_to_structured(records, dtype=dtype)

        # Save table to file
        np.savetxt(name, table, fmt=fmt, delimiter=delim, header=header, 
            comments='')

    def exportFlatFileCopy(self, name):
        ''' Create a new flat file based on the loaded flat file 
            using the latest data
        '''
        # Build records
        records, labels = self.getLatestRecords()

        # Write data to flat file
        ff = ff_writer(name, copy_header=self.FIDs[0].name)
        ff.set_data(records[:,0], records[:,1:])
        ff.write()

    def getLatestRecords(self):
        ''' 
            Returns the array of records in the data, using the latest
            edited data
        '''

        # Assemble list of unique time arrays and their corresponding data arrays
        timeDict = {} # Index -> (dstr, data) list
        timeMap = {} # Index -> Times
        index = 0
        for dstr in self.DATADICT:
            times = self.getTimes(dstr, self.currentEdit)[0]
            data = self.getData(dstr, self.currentEdit)

            # If time array previously seen, add to its timeDict list
            reusedTimes = False
            for key in timeMap:
                if np.array_equal(timeMap[key], times):
                    reusedTimes = True
                    timeDict[key].append((dstr, data))
                    break

            # Create new entry in dictionary
            if not reusedTimes:
                timeDict[index] = [(dstr, data)]
                timeMap[index] = times
                index += 1
        del key
        del index

        # Find longest time array
        lastKey = None
        for key in timeMap:
            if lastKey is None or len(timeMap[key]) > len(timeMap[lastKey]):
                lastKey = key
        del key
        fullTimes = timeMap[lastKey]

        # Fill all other time arrays and data arrays to be same length
        # as fullTimes
        fullData = [fullTimes]
        labels = []
        for key in timeDict:
            # Get relative indices and create filler arrays if time range is
            # only a subset of full times
            currTimes = timeMap[key]
            sI = bisect.bisect_right(fullTimes, currTimes[0])
            if currTimes[0] == fullTimes[0]:
                sI = 0
            eI = bisect.bisect_right(fullTimes, currTimes[-1])
            startFill = [self.errorFlag]*sI
            endFill = [self.errorFlag]*(max(len(fullTimes) - eI - 1, 0))

            # Concatenate filler arrays with each data array
            for dstr, data in timeDict[key]:
                newData = np.concatenate([startFill, data, endFill])
                fullData.append(newData)
                labels.append(dstr)

        # Stack full dataset
        records = np.stack(fullData, axis=1)

        return records, labels

    def updateTraceStats(self):
        if self.tools['Stats']:
            self.tools['Stats'].update()

    def updateCurlometer(self):
        if self.tools['Curlometer']:
            self.tools['Curlometer'].calculate()

    def updateDynCohPha(self):
        if self.tools['DynCohPha']:
            self.tools['DynCohPha'].updateParameters()

    def updateDynamicSpectra(self):
        if self.tools['DynSpectra']:
            self.tools['DynSpectra'].updateParameters()

    def updateDynWave(self):
        if self.tools['DynWave']:
            self.tools['DynWave'].updateParameters()

    # color is hex string ie: '#ff0000' for red
    def initGeneralSelect(self, name, color, timeEdit, mode, startFunc=None, updtFunc=None, 
        closeFunc=None, canHide=False, maxSteps=1):
        self.endGeneralSelect()
        self.closeTimeSelect()

        if timeEdit is not None:
            timeEdit.linesConnected = False
        self.currSelect = GeneralSelect(self, mode, name, color, timeEdit,
            func=startFunc, updtFunc=updtFunc, closeFunc=closeFunc, maxSteps=maxSteps)

        # Enable selection menu
        self.currSelect.setFullSelectionTrigger(self.updateSelectionMenu)

        # Apply a saved region if there is one
        if self.savedRegion:
            a, b = self.savedRegion.regions[0].getRegion()
            self.currSelect.addRegion(a-self.tickOffset,b-self.tickOffset)
        elif self.batchSelect:
            # If batch select is open and one of the following tools
            tools = ['Spectra', 'Dynamic Spectra', 'Dynamic Coh/Pha',
                'Wave Analysis', 'Detrend']
            if name in tools:
                abbrv = self.toolNameMap[name]
                # Auto-select an empty region
                self.currSelect.addRegion(0, 0, update=False)

                # Bring batch select to the top and set the function it will
                # call to update the tool window
                self.batchSelect.raise_()
                self.batchSelect.setUpdateInfo(timeEdit, self.tools[abbrv].update)
                self.batchSelect.update()

                # Show tool window
                self.tools[abbrv].show()

    def endGeneralSelect(self):
        if self.currSelect:
            self.currSelect.closeAllRegions()
            self.currSelect = None
            self.updateSelectionMenu()

        elif self.batchSelect:
            self.batchSelect.setUpdateInfo(None, None)

    # get slider ticks from time edit
    def getTicksFromTimeEdit(self, timeEdit):
        # Get datetimes and convert to ticks
        start = timeEdit.start.dateTime().toPyDateTime()
        end = timeEdit.end.dateTime().toPyDateTime()

        t0 = ff_time.date_to_tick(start, self.epoch)
        t1 = ff_time.date_to_tick(end, self.epoch)

        i0 = self.calcTickIndexByTime(t0)
        i1 = self.calcTickIndexByTime(t1)
        return (i0,i1) if i0 < i1 else (i1,i0) #need parenthesis here, otherwise it will eval like 3 piece tuple with if in the middle lol yikes

    # tick index refers to slider indices
    # THIS IS NOT ACCURATE when the time resolution is varying (which is usual)
    # keeping this one for now because not sure how to get function below this one to work when loading multiple files yet
    def calcTickIndexByTime(self, t):
        perc = (t - self.minTime) / (self.maxTime - self.minTime)
        perc = Mth.clamp(perc, 0, 1)
        assert perc >= 0 and perc <= 1
        return int(perc * self.iiE)

    # could make combo of above two functions
    # tries to use second function when it can (find correct times file) otherwise uses first
    # somehow needs to figure out which times the tick values are within or something
    def calcDataIndicesFromLines(self, dstr, editNumber, regNum=0):
        """given a data string, calculate its indices based on time range currently selected with lines"""

        times = self.getTimes(dstr,editNumber)[0]
        t0,t1 = self.getSelectionStartEndTimes(regNum)
        i0 = self.calcDataIndexByTime(times, t0)
        i1 = self.calcDataIndexByTime(times, t1)
        if i1 > len(times)-1:
            i1 = len(times)-1
        if i0 >= i1:
            i0 = i1-1
        return i0,i1

    def calcDataIndexByTime(self, times, t):
        return data_util.get_data_index(times, t)

    def getTimeTicksFromTimeEdit(self, timeEdit):
        return data_util.get_ticks_from_edit(timeEdit, self.epoch)

    def getTimestampFromTick(self, tick):
        return ff_time.tick_to_ts(tick, self.epoch)

    def datetime_from_tick(self, tick):
        ts = self.getTimestampFromTick(tick)
        fmt = '%Y %j %b %d %H:%M:%S.%f'
        date = datetime.strptime(ts, fmt)
        return date

    def getTickFromDateTime(self, dt):
        return ff_time.date_to_tick(dt, self.epoch)

    def getDateTimeObjFromTick(self, tick):
        return ff_time.tick_to_date(tick, self.epoch)
    
    def getTickFromTimestamp(self, ts):
        date = parser.isoparse(ts)
        tick = ff_time.date_to_tick(date, self.epoch)
        return tick
    
    def setTimeEditByTicks(self, t0, t1, timeEdit):
        dt0 = self.getDateTimeFromTick(t0)
        dt1 = self.getDateTimeFromTick(t1)
        timeEdit.start.setDateTime(dt0)
        timeEdit.end.setDateTime(dt1)

    def getSelectionStartEndTimes(self, regNum=0):
        if self.currSelect is None or self.currSelect.regions == []:
            return self.tO, self.tE
        t0, t1 = self.currSelect.regions[regNum].getRegion()
        return (t0,t1) if t0 <= t1 else (t1,t0) # need parens here!

    def getSelectedPlotInfo(self):
        """based on which plots have active lines, return list for each plot of the datastr and pen for each trace"""

        if self.currSelect is None or self.currSelect.regions == []:
            return []

        plotInfo = []
        region = self.currSelect.regions[0]
        plots = self.pltGrd.get_plots()
        for pltNum in range(0, len(region.regionItems)):
            if region.isVisible(pltNum) and not plots[pltNum].isSpecialPlot():
                plotInfo.append((self.lastPlotStrings[pltNum], self.plotTracePens[pltNum]))

        return plotInfo

    def findVecGroups(self):
        # Find groups of dstrs that match x,y,z pattern and organize them
        # by axis
        dstrs = self.DATASTRINGS[:]
        found = []
        for kw in ['BX','BY','BZ']:
            f = []
            for dstr in dstrs:
                if kw.lower() in dstr.lower():
                    f.append(dstr)
            found.append(f)
        return found
    
    def vecDict(self):
        vec_dict = {}
        grps = self.findPlottedVecGroups()
        for grp in grps:
            vec_dict[grp[0]] = grp
        return vec_dict

    def findPlottedVecGroups(self):
        # Try to identify fully plotted vectors by looking at the
        # identified axis groups and comparing them to the plotted strings
        plottedGrps = []

        # Get flattened list of currently plotted variables
        plottedDstrs = []
        for dstrLst in self.lastPlotStrings:
            for dstr, en in dstrLst:
                plottedDstrs.append(dstr)

        # Remove 'Bx' kw from list of 'x' axis dstrs and try to see if there
        # are other plotted dstrs w/ the same ending in the other axis groups
        grps = self.findVecGroups()
        firstRow = grps[0]
        for dstr in firstRow:
            if dstr not in plottedDstrs:
                continue

            matchingGrp = [dstr]
            strpDstr = dstr.strip('Bx').strip('BX').strip('bx')
            for rowGrp in grps[1:3]:
                for otherDstr in rowGrp:
                    if strpDstr in otherDstr and otherDstr in plottedDstrs:
                        matchingGrp.append(otherDstr)

            if len(matchingGrp) == 3: # Fully plotted vector
                plottedGrps.append(matchingGrp)

        return plottedGrps

    def autoSelectRange(self):
        # Automatically select the section currently being viewed
        t0, t1 = self.tO-self.tickOffset, self.tE-self.tickOffset
        if self.currSelect == None:
            self.startTool('Stats')
        self.currSelect.addRegion(t0, t1)
    
    def getCurrentTool(self, setTrace=True):
        if self.currSelect:
            return self.currSelect
        elif setTrace:
            self.startTool('Stats')
            return self.currSelect
        else:
            return None

    def gridLeftClick(self, x, vb, ctrlPressed):
        # Get current tool selection (or set trace as select tool if batch select
        # is not open) and pass the left click to it
        batchOpen = self.batchSelect is not None
        tool = self.getCurrentTool(setTrace=(not batchOpen))
        if tool:
            tool.leftClick(x, vb, ctrlPressed)

        # If batch select is open and the selections are not locked, then
        # pass the left click to it as well
        if batchOpen and not self.batchSelect.isLocked():
            tool = self.batchSelect.linkedRegion
            tool.leftClick(x, vb, ctrlPressed)

    def gridRightClick(self, vb):
        # Apply right click to current toolg
        tool = self.getCurrentTool(setTrace=False) # Don't set trace as select
        if tool:
            tool.rightClick(vb)

        # Return whether this click was applied to a selection or not so the viewbox
        # can determine whether to apply the default right click action instead
        res = True if tool is not None else False
        return res

def read_files(read_funcs, files, progress_func, cancel_func, *args, **kwargs):
    ''' Function used to facilitate reading of files asynchronously '''
    datas = []
    n = len(files)
    for i, (read_func, path) in enumerate(zip(read_funcs, files)):
        # Get filename and read function
        base = os.path.basename(path)

        # Update progress and return if cancel function is pressed        
        progress_func((base, i, n, ProgStates.LOADING))
        if cancel_func():
            return []

        # Read in data from files
        try:
            data = read_func(path, *args, **kwargs)
            datas.append(data)
            progress_func((base, i, n, ProgStates.SUCCESS))
        except:
            progress_func((base, i, n, ProgStates.FAILURE))

    # Do not return anything if thread is to be canceled
    if cancel_func():
        return []
    return datas

def read_cdf_files(files, progress_func, cancel_func, exclude_keys=None, 
        clip_range=None, label_funcs=None):
    ''' Function used to facilitate reading of CDF files asynchronously '''
    file_datas = []
    n = len(files)
    for i in range(len(files)):
        # Get filename and label function if given
        file = files[i]
        base = os.path.basename(file)
        label_func = None if label_funcs is None else label_funcs[i]

        # Update progress and return if cancel function is pressed        
        progress_func((base, i, n, ProgStates.LOADING))
        if cancel_func():
            return []

        # Read in data from files and create a tuple for each set of data
        # read from the CDF (since multiple epochs are allowed)
        datas, reader = load_cdf(file, exclude_keys, label_func=label_func, clip_range=clip_range)
        file_data = [(data, reader) for key, data in datas.items()]
        file_datas.extend(file_data)

    # Do not return anything if thread is to be canceled
    if cancel_func():
        return []
    
    return file_datas

def myexepthook(type, value, tb):
    print(f'{type} {value}')
    traceback.print_tb(tb,limit=5)
    os.system('pause')

def startMagPy(files=None, display=True):
    '''
    Main function for creating MagPy4Window object and starting program
    '''
    # Set up application
    app = QtWidgets.QApplication(sys.argv)

    app.setOrganizationName('IGPP UCLA')
    app.setOrganizationDomain('igpp.ucla.edu')
    app.setApplicationName('MagPy4')

    # Set fusion as default style if found
    keys = QtWidgets.QStyleFactory().keys()
    if 'Fusion' in keys:
        app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))

    # Create the MagPy4 window
    main = MagPy4Window(app)
    if display:
        main.showMaximized()

    # Add in fonts to database
    font_dir = get_relative_path('fonts')
    fonts = os.listdir(font_dir)
    fonts = [os.path.join(font_dir, f) for f in fonts]

    if os.name in ['posix', 'nt']:
        db = QtGui.QFontDatabase()
        for font in fonts:
            db.addApplicationFont(font)

    # Set default any default application fonts
    if os.name == 'nt':
        font = QtGui.QFont('Roboto')
        app.setFont(font)

    # Initialize any files passed
    if files is not None and len(files) > 0:
        split_name = files[0].split('.')
        extension = split_name[-1]
        if extension == 'cdf': # CDF
            main.addCDF(files, clearPrev=True)
        else:
            main.openFileList(files, True)

    if display:
        args = sys.argv
        sys.excepthook = myexepthook
        sys.exit(app.exec_())
    else:
        return main

def runMarsPy():
    runMagPy()

def runMagPy():
    # Read in arguments, opening MagPy if the update flag was not passed,
    # and passing along any the names of any files to open at startup
    res, files = readArgs()
    if res:
        startMagPy(files=files)

def updateMagPy():
    '''
    Runs command to install latest version of MagPy4 from GitHub through pip
    '''
    gitLink = 'git+https://github.com/igpp-ucla/MagPy4.git'

    # Determine which command to use based on system type
    windowsMode = (os.name == 'nt')
    updtCmd = f'pip3 install {gitLink}'
    if windowsMode:
        updtCmd = f'py -m pip install {gitLink}'

    # Run command
    os.system(updtCmd)

def readArgs():
    '''
    Reads commandline arguments and runs additional actions based on flags
    Returns a tuple;
        - A bool indicating whether the full program should be opened/ran
        - A list of filenames the user passed, if any; Otherwise, returns None
    '''
    args = setupParser()

    runFlag = True # Determines whether to run MagPy or not
    ffLst = None # A list of files to load at runtime, None if no filenames passed

    # Skip loading if updating MagPy; run update script instead
    if args.update:
        updateMagPy()
        runFlag = False

    # Check if any filenames were passed
    if args.ffName != []:
        ffLst = []
        for ffPath in args.ffName:
            # If path is relative, replace with the absolute path
            if not os.path.isabs(ffPath):
                ffPath = os.path.join(os.getcwd(), ffPath)
            ffLst.append(ffPath)

    return runFlag, ffLst

def setupParser():
    '''
    Sets up command-line arguments for MagPy script
    '''
    desc = 'Magnetic Field Analysis Program'
    parser = argparse.ArgumentParser(description=desc)

    # Update argument
    updtHlp = 'Updates MagPy to the latest version from GitHub'
    parser.add_argument('--update', help=updtHlp, action='store_true')

    # Add option to pass in a list of flat file names
    info = 'Flat File(s) to load at start up'
    parser.add_argument('ffName', help=info, metavar='FF', type=str, const=None, 
        nargs='*')

    # Results of parsed arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    runMagPy()