"""
Main module for the program
handles data, plotting and main window management
"""

# python 3.6
import os
import sys
import pickle
import argparse

# so python looks in paths for these folders too
# maybe make this into actual modules in future
sys.path.insert(0, 'ffPy')
sys.path.insert(0, 'cdfPy')

# Version number and copyright notice displayed in the About box
NAME = f'MagPy4'
VERSION = f'Version 1.4.7.0 (July 21, 2020)'
COPYRIGHT = f'Copyright © 2020 The Regents of the University of California'

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
import pyqtgraph as pg

import FF_File
from FF_Time import FFTIME, leapFile

from .MagPy4UI import MagPy4UI, PyQtUtils, MainPlotGrid, StackedLabel, TimeEdit, StackedAxisLabel, FileLabel
from .pyqtgraphExtensions import TrackerRegion
from .plotMenu import PlotMenu
from .spectra import Spectra
from .dataDisplay import DataDisplay, UTCQDate
from .plotAppearance import MagPyPlotApp
from .addTickLabels import AddTickLabels
from .edit import Edit
from .traceStats import TraceStats
from .helpWindow import HelpWindow
from .AboutDialog import AboutDialog
from .plotBase import DateAxis, MagPyPlotItem, MagPyPlotDataItem
from .MMSTools import PlaneNormal, Curlometer, Curvature, ElectronPitchAngle, ElectronOmni, PressureTool, FEEPS_EPAD
from . import mms_orbit
from . import mms_formation
from .dynBase import SimpleColorPlot
from .detrendWin import DetrendWindow
from .ASCII_Importer import Asc_Importer, ASC_Output
from .dynamicSpectra import DynamicSpectra, DynamicCohPha
from .waveAnalysis import DynamicWave
from .trajectory import TrajectoryAnalysis
from .smoothingTool import SmoothingTool
from .ffCreator import createFF
from .mth import Mth
from .tests import Tests
import bisect
from .timeManager import TimeManager
from .selectionManager import GeneralSelect, FixedSelection, TimeRegionSelector, BatchSelect, SelectableViewBox
from .layoutTools import BaseLayout

import time
import functools
import multiprocessing as mp
import traceback

from . import getRelPath # Function to get path to use to refer to data/img files

CANREADCDFS = False

class MagPy4Window(QtWidgets.QMainWindow, MagPy4UI, TimeManager):
    def __init__(self, app, parent=None):
        super(MagPy4Window, self).__init__(parent)
        TimeManager.__init__(self, 0, 0, None)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True) #todo add option to toggle this
        #pg.setConfigOption('useOpenGL', True)

        self.app = app
        self.ui = MagPy4UI()
        self.ui.setupUI(self)

        global CANREADCDFS
        self.ui.actionOpenCDF.setDisabled(not CANREADCDFS)

        self.OS = os.name
        if os.name == 'nt':
            self.OS = 'windows'
        print(f'OS: {self.OS}')

        self.ui.scrollSelect.startChanged.connect(self.onStartSliderChanged)
        self.ui.scrollSelect.endChanged.connect(self.onEndSliderChanged)
        self.ui.scrollSelect.rangeChanged.connect(self.rangeChanged)
        self.ui.scrollSelect.sliderReleased.connect(self.setTimes)

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

        self.ui.timeEdit.start.dateTimeChanged.connect(self.onStartEditChanged)
        self.ui.timeEdit.end.dateTimeChanged.connect(self.onEndEditChanged)

        # Scrolling zoom connects
        self.ui.scrollPlusShrtct.activated.connect(self.increasePlotHeight)
        self.ui.scrollMinusShrtct.activated.connect(self.decreasePlotHeight)

        # Toggle tracker shortcuts/actions
        self.ui.toggleTrackerAction.triggered.connect(self.toggleTracker)

        # Main menu action connections
        self.ui.actionOpenFF.triggered.connect(functools.partial(self.openFileDialog, True,True, None))
        self.ui.actionAddFF.triggered.connect(functools.partial(self.openFileDialog, True, False, None))
        self.ui.actionOpenASCII.triggered.connect(functools.partial(self.openFileDialog, False, True, None))

        self.ui.actionExportDataFile.triggered.connect(self.exportFile)

        self.ui.actionExportFF.triggered.connect(self.exportFlatFile)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionOpenWs.triggered.connect(self.openWsOpenDialog)
        self.ui.actionSaveWs.triggered.connect(self.openWsSaveDialog)
        self.ui.actionShowData.triggered.connect(self.showData)
        self.ui.actionPlotMenu.triggered.connect(self.openPlotMenu)
        self.ui.actionSpectra.triggered.connect(self.startSpectra)
        self.ui.actionDynamicSpectra.triggered.connect(self.startDynamicSpectra)
        self.ui.actionDynamicCohPha.triggered.connect(self.startDynamicCohPha)
        self.ui.actionDynWave.triggered.connect(self.startDynWave)
        self.ui.actionTraj.triggered.connect(self.openTraj)
        self.ui.actionDetrend.triggered.connect(self.startDetrend)
        self.ui.actionEdit.triggered.connect(self.openEdit)
        self.ui.actionHelp.triggered.connect(self.openHelp)
        self.ui.actionAbout.triggered.connect(self.openAbout)
        self.ui.switchMode.triggered.connect(self.swapMode)
        self.ui.runTests.triggered.connect(self.runTests)
        self.ui.actionBatchSelect.triggered.connect(self.openBatchSelect)
        self.ui.actionChkForUpdt.triggered.connect(self.checkForUpdate)

        # MMS Tool actions
        self.ui.actionPlaneNormal.triggered.connect(self.openPlaneNormal)
        self.ui.actionCurlometer.triggered.connect(self.openCurlometer)
        self.ui.actionCurvature.triggered.connect(self.openCurvature)
        self.ui.actionEPAD.triggered.connect(self.startEPAD)
        self.ui.actionEOmni.triggered.connect(self.startEOMNI)
        self.ui.actionFEEPSPAD.triggered.connect(self.startFEEPSEPAD)
        self.ui.actionMMSPressure.triggered.connect(self.openPressure)
        self.ui.actionMMSOrbit.triggered.connect(self.openMMSOrbit)
        self.ui.actionMMSFormation.triggered.connect(self.openMMSFormation)

        # Selection menu actions
        self.ui.actionFixSelection.triggered.connect(self.fixSelection)
        self.ui.actionSelectByTime.triggered.connect(self.openTimeSelect)
        self.ui.actionSelectView.triggered.connect(self.autoSelectRange)

        # Content menu action connections
        self.ui.plotApprAction.triggered.connect(self.openPlotAppr)
        self.ui.addTickLblsAction.triggered.connect(self.openAddTickLbls)

        # options menu dropdown
        self.ui.scaleYToCurrentTimeAction.toggled.connect(self.updateYRange)
        self.ui.antialiasAction.toggled.connect(self.toggleAntialiasing)
        self.ui.bridgeDataGaps.toggled.connect(self.replotDataCallback)
        self.ui.drawPoints.toggled.connect(self.replotDataCallback)
        self.ui.downsampleAction.toggled.connect(self.enableDownsampling)
        self.ui.showFileLbl.toggled.connect(self.showFileLabel)
        self.ui.enableScrollingAction.toggled.connect(self.enableScrolling)

        # Disable the Tools and Options menus. They'll be enabled after the user opens a file.
        self.DATASTRINGS = []
        self.enableToolsAndOptionsMenus(False)

        self.plotAppr = None
        self.addTickLbls = None
        self.helpWindow = None
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
        self.fileNameLabel = None
        self.minimumPlotHeight = 3 # Inches

        self.toolNames = ['Data', 'Edit', 'Plot Menu', 'Detrend', 'Spectra',
            'Dynamic Spectra', 'Dynamic Coh/Pha', 'Wave Analysis',
            'Trajectory Analysis', 'Stats', 'Electron/Ion Spectrum', 'FEEPS EPAD',
            'Electron PAD', 'Plane Normal', 'Curlometer', 'Curvature',
            'MMS Orbit', 'MMSFormation', 'MMS Pressure']

        self.toolAbbrv = ['Data', 'Edit', 'PlotMenu', 'Detrend', 'Spectra',
            'DynSpectra', 'DynCohPha', 'DynWave', 'Traj', 'Stats',
            'EOMNI', 'FEEPS EPAD', 'EPAD', 'PlaneNormal', 'Curlometer', 
            'Curvature', 'MMSOrbit', 'MMSFormation', 'Pressure']

        self.toolInitFuncs  = [self.showData, self.openEdit, self.openPlotMenu,
            self.startDetrend, self.startSpectra, self.startDynamicSpectra,
            self.startDynamicCohPha, self.startDynWave, self.openTraj,
            self.startTraceStats, self.startEOMNI, self.startFEEPSEPAD, self.startEPAD,
            self.openPlaneNormal, self.openCurlometer, self.openCurlometer,
            self.openMMSOrbit, self.openMMSFormation, self.openPressure]

        self.tools = {}
        self.toolNameMap = {}
        self.toolFuncs = {}
        for name, abbrv, f in zip(self.toolNames, self.toolAbbrv, self.toolInitFuncs):
            self.toolNameMap[name] = abbrv
            self.tools[abbrv] = None
            self.toolFuncs[abbrv] = f

        # these are saves for options for program lifetime
        self.plotMenuTableMode = True
        self.traceStatsOnTop = True
        self.mouseEnabled = False

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
        img_path = getRelPath('images', directory=True)
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
        winWidth = abs(self.iE - self.iO) # Number of ticks currently displayed
        return winWidth

    def setNewWindowTicks(self, newTO, newTE):
        # Update slider values and self.iO, self.iE
        self.setSliderNoCallback('start', newTO)
        self.setSliderNoCallback('stop', newTE)

        # Update timeEdit values
        self.onStartSliderChanged(newTO)
        self.onEndSliderChanged(newTE)
        self.ui.timeEdit.start.update() # Update appearance for OSX users
        self.ui.timeEdit.end.update()

        # Update plots
        self.setTimes()

    def shiftWindow(self, direction):
        winWidth = self.getWinTickWidth() # Number of ticks currently displayed
        shiftAmt = int(winWidth*self.shftPrcnt)

        if direction == 'L': # Shift amt is negative if moving left
            shiftAmt = shiftAmt * (-1) 
        newTO = self.iO + shiftAmt
        newTE = self.iE + shiftAmt

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
        if (origin < 0):
            origin = 0
            end = origin + winWidth
        elif end > self.iiE:
            end = self.iiE
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

    def viewAllData(self):
        self.setNewWindowTicks(0, self.iiE)

    # Use these two functions to set a temporary status msg and clear it
    def showStatusMsg(self, msg):
        status = 'STATUS: ' + msg
        self.ui.statusBar.showMessage(status)

    def clearStatusMsg(self):
        self.ui.statusBar.clearMessage()

    def showFileLabel(self, val):
        if self.fileNameLabel:
            self.fileNameLabel.setVisible(val)

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
        if self.pltGrd:
            self.pltGrd.deleteLater()

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
        self.closePlotMenu()
        self.closePlotAppr()
        self.closeEdit()
        self.closeData()
        self.closeTraceStats()
        self.closeSpectra()
        self.closeSmoothing()
        self.closeAddTickLbls()
        self.closePlaneNormal()
        self.closeDynamicSpectra()
        self.closeDynamicCohPha()
        self.closeDynWave()
        self.closeMMSTools()
        self.closeDetrend()
        self.closeTraj()
        self.closeFixSelection()
        self.closeTimeSelect()
        self.closeBatchSelect()
        self.closeMMSOrbit()
        self.closeMMSFormation()

    def closePlotTools(self):
        self.closeDetrend()
        self.closeSpectra()
        self.closeDynamicCohPha()
        self.closeDynamicSpectra()
        self.closeDynWave()
        self.closeTraceStats()
        self.closeTraj()
        self.closeMMSTools()

    def initVariables(self):
        """init variables here that should be reset when file changes"""
        self.lastPlotStrings = None
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
        if self.helpWindow:
            self.helpWindow.close()
            self.helpWindow = None

    def closeAbout(self):
        if self.aboutDialog:
            self.aboutDialog.close()
            self.aboutDialog = None

    def closeMMSTools(self):
        self.closePlaneNormal()
        self.closeCurlometer()
        self.closeCurvature()
        self.closeEPAD()
        self.closeEOMNI()
        self.closeFEEPSEPAD()
        self.closeMMSFormation()
        self.closeMMSOrbit()
        self.closePressure()

    def startEOMNI(self):
        self.closeEOMNI()
        if self.tools['EOMNI'] is None or self.tools['EOMNI'].wasClosed:
            self.tools['EOMNI'] = ElectronOmni(self)
            self.initGeneralSelect('Electron/Ion Spectrum', None, self.tools['EOMNI'].ui.timeEdit,
            'Single', self.showEOMNI, closeFunc=self.closeEOMNI)

    def showEOMNI(self):
        if self.tools['EOMNI']:
            self.tools['EOMNI'].show()
            self.tools['EOMNI'].update()

    def closeEOMNI(self):
        if self.tools['EOMNI']:
            self.tools['EOMNI'].close()
            self.tools['EOMNI'] = None

    def startEPAD(self):
        self.closeEPAD()
        if self.tools['EPAD'] is None or self.tools['EPAD'].wasClosed:
            self.tools['EPAD'] = ElectronPitchAngle(self)
            self.initGeneralSelect('Electron PAD', '#0a22ff', self.tools['EPAD'].ui.timeEdit, 
            'Single', self.showEPAD, closeFunc=self.closeEPAD)

    def closeEPAD(self):
        if self.tools['EPAD']:
            self.tools['EPAD'].close()
            self.tools['EPAD'] = None
    
    def showEPAD(self):
        if self.tools['EPAD']:
            self.clearStatusMsg()
            self.tools['EPAD'].show()
            self.tools['EPAD'].update()

    def startFEEPSEPAD(self):
        self.closeFEEPSEPAD()
        self.tools['FEEPS EPAD'] = FEEPS_EPAD(self)
        self.initGeneralSelect('FEEPS PAD', '#000000', self.tools['FEEPS EPAD'].ui.timeEdit,
            'Single', self.showFEEPSPAD, closeFunc=self.closeFEEPSEPAD)
    
    def showFEEPSPAD(self):
        if self.tools['FEEPS EPAD']:
            self.clearStatusMsg()
            self.tools['FEEPS EPAD'].show()
            self.tools['FEEPS EPAD'].update()

    def closeFEEPSEPAD(self):
        if self.tools['FEEPS EPAD']:
            self.tools['FEEPS EPAD'].close()
            self.tools['FEEPS EPAD'] = None

    def openCurlometer(self):
        self.closeMMSTools()
        self.tools['Curlometer'] = Curlometer(self)
        self.initGeneralSelect('Curlometer', '#ffa500', self.tools['Curlometer'].ui.timeEdit, 
            'Adjusting', self.showCurlometer, self.updateCurlometer, 
            closeFunc=self.closeCurlometer, maxSteps=-1)

    def showCurlometer(self):
        if self.tools['Curlometer']:
            self.tools['Curlometer'].show()
            self.tools['Curlometer'].calculate()

    def closeCurlometer(self):
        if self.tools['Curlometer']:
            self.tools['Curlometer'].close()
            self.tools['Curlometer'] = None

    def openCurvature(self):
        self.closeMMSTools()
        self.tools['Curvature'] = Curvature(self)
        self.initGeneralSelect('Curvature', '#ff4242', self.tools['Curvature'].ui.timeEdit,
            'Single', self.showCurvature, closeFunc=self.closeCurvature)

    def showCurvature(self):
        if self.tools['Curvature']:
            self.tools['Curvature'].show()
            self.tools['Curvature'].update()

    def closeCurvature(self):
        if self.tools['Curvature']:
            self.endGeneralSelect()
            self.tools['Curvature'].close()

    def openPlaneNormal(self):
        self.closeMMSTools()
        self.tools['PlaneNormal'] = PlaneNormal(self)
        self.initGeneralSelect('Plane Normal', '#42f495', None, 'Single',
            startFunc=self.showNormal, closeFunc=self.closePlaneNormal)

    def showNormal(self):
        if self.tools['PlaneNormal']:
            self.tools['PlaneNormal'].show()
            self.tools['PlaneNormal'].update()

    def closePlaneNormal(self):
        if self.tools['PlaneNormal']:
            self.tools['PlaneNormal'].close()
            self.tools['PlaneNormal'] = None
            self.endGeneralSelect()

    def openPressure(self):
        self.closePressure()
        self.tools['Pressure'] = PressureTool(self)
        self.initGeneralSelect('Pressure', '#9200EE', 
            self.tools['Pressure'].ui.timeEdit, 'Single', 
            startFunc=self.showPressure, closeFunc=self.closePressure)

    def showPressure(self):
        if self.tools['Pressure']:
            self.tools['Pressure'].show()
            self.tools['Pressure'].update()

    def closePressure(self):
        if self.tools['Pressure']:
            self.tools['Pressure'].close()
            self.tools['Pressure'] = None

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

    def openPlotMenu(self):
        self.closePlotMenu()
        self.tools['PlotMenu'] = PlotMenu(self)

        geo = self.geometry()
        self.tools['PlotMenu'].move(geo.x() + 200, geo.y() + 100)
        self.tools['PlotMenu'].show()

    def openPlotAppr(self):
        self.closePlotAppr()
        self.plotAppr = MagPyPlotApp(self, self.plotItems)
        self.plotAppr.show()

    def closePlotAppr(self):
        if self.plotAppr:
            self.plotAppr.close()
            self.plotAppr = None

    def openAddTickLbls(self):
        self.closeAddTickLbls()
        self.addTickLbls = AddTickLabels(self, self.pltGrd)
        self.addTickLbls.show()

    def closeAddTickLbls(self):
        if self.addTickLbls:
            self.addTickLbls.close()
            self.addTickLbls = None

    def openEdit(self):
        self.closeTraceStats()
        self.closeEdit()
        self.tools['Edit'] = Edit(self)
        self.tools['Edit'].show()

    def openTraj(self):
        self.closeTraceStats()
        self.closeTraj()
        self.tools['Traj'] = TrajectoryAnalysis(self)

        if not self.tools['Traj'].validState():
            self.tools['Traj'].close()
            self.tools['Traj'] = None
            errMsg = 'Error: Cannot open trajectory analysis window without position data'
            self.ui.statusBar.showMessage(errMsg)
            return

        self.initGeneralSelect('Trajectory Analysis', '#34ebdc', self.tools['Traj'].ui.timeEdit, 'Single',
            self.startTraj, closeFunc=self.closeTraj)

    def startTraj(self):
        if self.tools['Traj']:
            self.tools['Traj'].show()
            self.tools['Traj'].ui.altFrame.updatePlot()
            self.tools['Traj'].ui.orbitFrame.updatePlot()
    
    def closeTraj(self):
        if self.tools['Traj'] is not None:
            self.tools['Traj'].close()
            self.tools['Traj'] = None

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
            self.savedRegion.closeAllRegions()
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
            self.startTraceStats()

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

    def openTraceStats(self):
        if self.tools['Stats']:
            self.tools['Stats'].show()

    def startTraceStats(self):
        self.closePlotTools()
        self.closeTraceStats()
        self.tools['Stats'] = TraceStats(self)
        self.initGeneralSelect('Stats', None, self.tools['Stats'].ui.timeEdit,
            'Adjusting', self.openTraceStats, self.updateTraceStats, closeFunc=self.closeTraceStats, 
            maxSteps=-1)

    def startDynamicSpectra(self):
        self.closePlotTools()
        if not self.tools['DynSpectra'] or self.tools['DynSpectra'].wasClosed:
            self.tools['DynSpectra'] = DynamicSpectra(self)
            self.initGeneralSelect('Dynamic Spectra', '#c700ff', self.tools['DynSpectra'].ui.timeEdit,
                'Single', self.showDynamicSpectra, self.updateDynamicSpectra,
                closeFunc=self.closeDynamicSpectra)
            self.showStatusMsg('Selecting dynamic spectrogram range...')

    def showDynamicSpectra(self):
        if self.tools['DynSpectra']:
            self.clearStatusMsg()
            self.tools['DynSpectra'].show()
            self.tools['DynSpectra'].update()

    def closeDynamicSpectra(self):
        if self.tools['DynSpectra']:
            self.clearStatusMsg()
            self.tools['DynSpectra'].close()
            self.tools['DynSpectra'] = None

    def startDynamicCohPha(self):
        self.closePlotTools()
        if not self.tools['DynCohPha'] or self.tools['DynCohPha'].wasClosed:
            self.tools['DynCohPha'] = DynamicCohPha(self)
            self.showStatusMsg('Selecting dynamic coherence/phase range...')
            self.initGeneralSelect('Dynamic Coh/Pha', '#c551ff', self.tools['DynCohPha'].ui.timeEdit,
                'Single', self.showDynamicCohPha, self.updateDynCohPha, 
                closeFunc=self.closeDynamicCohPha)

    def showDynamicCohPha(self):
        if self.tools['DynCohPha']:
            self.clearStatusMsg()
            self.tools['DynCohPha'].show()
            self.tools['DynCohPha'].update()

    def closeDynamicCohPha(self):
        if self.tools['DynCohPha']:
            self.clearStatusMsg()
            self.tools['DynCohPha'].close()
            self.tools['DynCohPha'] = None

    def startSpectra(self):
        self.closePlotTools()
        if not self.tools['Spectra'] or self.tools['Spectra'].wasClosed:
            self.tools['Spectra'] = Spectra(self)
            spectra = self.tools['Spectra']
            self.showStatusMsg('Selecting spectra range...')
            self.initGeneralSelect('Spectra', '#c551ff', spectra.ui.timeEdit,
                'Single', self.showSpectra, closeFunc=self.closeSpectra)

    def showSpectra(self):
        if self.tools['Spectra']:
            self.clearStatusMsg()
            self.tools['Spectra'].show()
            self.tools['Spectra'].update()
            PyQtUtils.moveToFront(self.tools['Spectra'])
            QtCore.QTimer.singleShot(100, self.tools['Spectra'].setAspect)

    def startDynWave(self):
        self.closeDynWave()
        if not self.tools['DynWave'] or self.tools['DynWave'].wasClosed:
            self.tools['DynWave'] = DynamicWave(self)
            self.tools['DynWave'].showPreSelectWin()
            self.initGeneralSelect('Wave Analysis', None, self.tools['DynWave'].ui.timeEdit,
                'Single', self.showDynWave, self.updateDynWave,
                closeFunc=self.closeDynWave)

    def showDynWave(self):
        if self.tools['DynWave']:
            self.tools['DynWave'].show()
            self.tools['DynWave'].setUserSelections()
            self.tools['DynWave'].update()

    def closeDynWave(self):
        if self.tools['DynWave']:
            self.tools['DynWave'].close()
            self.tools['DynWave'] = None

    def startDetrend(self):
        self.closePlotTools()
        self.tools['Detrend'] = DetrendWindow(self)
        self.showStatusMsg('Selecting region of data to detrend...')
        self.initGeneralSelect('Detrend', '#00d122', self.tools['Detrend'].ui.timeEdit,
            'Single', self.showDetrend, closeFunc=self.closeDetrend)

    def showDetrend(self):
        if self.tools['Detrend']:
            self.clearStatusMsg()
            self.tools['Detrend'].update()
            self.tools['Detrend'].show()

    def closeDetrend(self):
        if self.tools['Detrend']:
            self.tools['Detrend'].close()
            self.tools['Detrend'] = None

    def startSmoothing(self):
        self.closeSmoothing()
        if self.tools['Edit']:
            self.tools['Edit'].showMinimized()
            self.smoothing = SmoothingTool(self, self.tools['Edit'])
            self.smoothing.restartSelect()
            self.smoothing.show()

    def closeSmoothing(self):
        if self.smoothing:
            self.smoothing.close()
            self.smoothing = None
            self.endGeneralSelect()
    
    def openHelp(self):
        self.closeHelp()
        self.helpWindow = HelpWindow(self)
        self.helpWindow.show()

    def openAbout(self):
        self.aboutDialog = AboutDialog(NAME, VERSION, COPYRIGHT, self)
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

    def exportFlatFile(self):
        # Get file name from user input
        filename = self.saveFileDialog('.ffd', 'Flat File', False)
        if filename is None:
            return

        # Get one of the loaded flat file IDs (TODO: Support for multiple files?)
        FID = self.FIDs[0]
        FID.open()

        # Extract the original flat file's column headers, units, and source info
        labels = FID.getColumnDescriptor('NAME')[1:]
        units = FID.getColumnDescriptor('UNITS')[1:]
        sources = FID.getColumnDescriptor('SOURCE')[1:]

        # Get the epoch used for this flat file
        epoch = FID.getEpoch()

        # Extract all data for current edit (ignoring added variables for now)
        dta = []
        for dstr in self.DATASTRINGS:
            if dstr in list(self.ORIGDATADICT.keys()):
                dta.append(self.getData(dstr, self.currentEdit))

        # Verify that the data all have same length
        dtaLens = list(map(len, dta))
        if min(dtaLens) != max(dtaLens):
            print ('Error: Data columns have different lengths')
            return

        # Transpose data, get times, and create the flat file
        dta = np.array(dta).T
        times = self.getTimes(dstr, self.currentEdit)[0]
        if len(times) != len(dta):
            print ('Error: Data length != times length')
            return
        createFF(filename, times, dta, labels, units, sources, epoch)

    def saveFileDialog(self, defSfx='.txt', defFilter='TXT file', appendSfx=True):
        defaultSfx = defSfx
        defFilter = defFilter + '(*'+defSfx+')'
        QQ = QtGui.QFileDialog(self)
        QQ.setAcceptMode(QtGui.QFileDialog.AcceptSave)
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

    def openFileDialog(self, isFlatfile, clearCurrent, ascState=None):
        fileNames = []
        if isFlatfile:
            fileNames = QtWidgets.QFileDialog.getOpenFileNames(self, caption="Open Flat File", options = QtWidgets.QFileDialog.ReadOnly, filter='Flat Files (*.ffd)')[0]
        else:
            fileNames = QtWidgets.QFileDialog.getOpenFileNames(self, caption="Open ASCII File", options = QtWidgets.QFileDialog.ReadOnly, filter='ASCII Files (*.csv *.tsv *.txt)')[0]

        if len(fileNames) < 0 or not self.validFilenames(fileNames):
            return
        
        if isFlatfile:
            self.openFileList(fileNames, isFlatfile, clearCurrent)
        else:
            self.openAscDialog(fileNames, ascState)

    def validFilenames(self, fileNames):
        for fileName in fileNames:
            if '.' not in fileName: # lazy extension check
                print(f'Bad file found, cancelling open operation')
                return False
        if not fileNames:
            print(f'No files selected, cancelling open operation')
            return False

        return True

    def openFileList(self, fileNames, isFlatfile, clearCurrent):
        if clearCurrent: # Clear previously opened files
            for fid in self.FIDs:
                fid.close()
            self.FIDs = []
            self.initDataStorageStructures()

        res = True
        for i in range(0, len(fileNames)):
            fileName = fileNames[i]
            if isFlatfile:        
                fileName = fileName.rsplit(".", 1)[0] #remove extension
                res = res and self.openFF(fileName)
            else:
                res = res and self.openTextFile(fileName)

        if res is False: # If a file was not successfully opened, return
            self.ui.statusBar.showMessage('Error: Could not open file', 5000)
            return res

        # Close any opened ASCII file format dialog windows
        self.closeAscDialog() 

        # Update other state information and finish setup
        self.finishOpenFileSetup()

        return res

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
        self.workspace = None

    def openAscDialog(self, filenames, ascState=None):
        # Open a dialog box to specify settings for opening this ASCII file
        # and link its apply btn to the openTextFile function
        self.closeAscDialog()
        self.asc = Asc_Importer(self, filenames[-1], self)

        # If opening from user-interface, open dialog
        if ascState is None:
            openFunc = functools.partial(self.openFileList, filenames, False, True)
            self.asc.linkApplyBtn(openFunc)
            self.asc.show()

            # Bring to front
            self.asc.activateWindow()
        else: # Otherwise load file state info into dialog and open files directly
            self.asc.loadStateInfo(ascState)

    def closeAscDialog(self):
        if self.asc:
            self.asc.close()
            self.asc = None

    def openTextFile(self, filename):
        # Set current file for ascii importer
        self.asc.setFile(filename)

        # Try to read in times, data, and other file info
        try:
            times, data, info = self.asc.readFile()
        except:
            self.ui.statusBar.showMessage('Error: Could not open ASCII file')
            return False

        # Extract other file info from tuple
        labels, units, epoch, errFlag, fd = info

        # Set error flag, epoch, and calculate a new resolution
        self.errorFlag = self.asc.getErrorFlag()
        self.epoch = epoch
        res = np.median(np.diff(times))
        self.resolution = min(self.resolution, res)

        # Load data into appropriate structures
        res = self.loadData(times, labels, data, units)
        if res is None:
            return False

        # Update FD list
        self.FIDs.append(fd)
        self.updateAfterOpening()
        return True

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

    def setPlotAttrs(self, pi):
        # add horizontal zero line
        zeroLine = pg.InfiniteLine(movable=False, angle=0, pos=0)
        zeroLine.setPen(pg.mkPen('#000000', width=1, style=QtCore.Qt.DotLine))
        pi.addItem(zeroLine, ignoreBounds=True)

        pi.hideButtons() # hide autoscale button

        # show top and right axis, but hide labels (they are off by default apparently)
        la = pi.getAxis('left')
        la.style['textFillLimits'] = [(0,1.1)] # no limits basically to force labels by each tick no matter what
        #la.setWidth(50) # this also kinda works but a little space wasteful, saving as reminder incase dynamic solution messes up

        ba = pi.getAxis('bottom')
        #ba.style['textFillLimits'] = [(0,1.1)]
        ta = pi.getAxis('top')
        ra = pi.getAxis('right')
        ta.show()
        ra.show()
        ta.setStyle(showValues=False)
        ra.setStyle(showValues=False)

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
        diff = np.diff(t)
        dta = dta[mask]
        times = (t, diff, res)

        self.TIMES.append(times)
        self.TIMEINDEX[dstr] = len(self.TIMES) - 1

        # Add in data to dictionaries, no units
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
        self.TIMES = [] # list of time informations (3 part lists) [time series, resolutions, average res]
        self.TIMEINDEX = {} # dict mapping dstrs to index into times list
        self.EDITEDTIMES = {}

        self.minTime = None # minimum time tick out of all loaded times
        self.maxTime = None # maximum
        self.iiE = None # maximum tick of sliders (min is always 0)
        self.resolution = 1000000.0 # this is set to minumum resolution when files are loaded so just start off as something large
        self.colorPlotInfo = {}

    def openFF(self, PATH):  # slot when Open pull down is selected
        FID = FF_File.FF_ID(PATH, status=FF_File.FF_STATUS.READ | FF_File.FF_STATUS.EXIST)
        if not FID:
            print('BAD FLATFILE')
            self.enableToolsAndOptionsMenus(False)
            return False
        err = FID.open()
        if err < 0:
            print('UNABLE TO OPEN')
            self.enableToolsAndOptionsMenus(False)
            return False

        print(f'\nOPEN {FID.name}')

        self.epoch = FID.getEpoch()
        print(f'epoch: {self.epoch}')
        #info = FID.FFInfo
        # errorFlag is usually 1e34 but sometimes less. still huge though
        self.errorFlag = FID.FFInfo['ERROR_FLAG'].value
        self.errorFlag = 1e16 # overriding for now since the above line is sometimes wrong depending on the file (i think bx saves as 1e31 but doesnt update header)
        print(f'error flag: {self.errorFlag:.0e}') # not being used currently
        #self.errorFlag *= 0.9 # based off FFSpectra.py line 829
        
        # load flatfile
        nRows = FID.getRows()
        records = FID.DID.sliceArray(row=1, nRow=nRows)
        ffTime = records["time"]
        dataByRec = records["data"]
        dataByCol = FF_File.arrayToColumns(records["data"])

        numRecords = len(dataByRec)
        numColumns = len(dataByCol)
        print(f'number records: {numRecords}')
        print(f'number columns: {numColumns}')

        datas = [np.array(col) for col in dataByCol]

        # ignoring first column because that is time, hence [1:]
        newDataStrings = FID.getColumnDescriptor("NAME")[1:]
        units = FID.getColumnDescriptor("UNITS")[1:]

        self.resolution = min(self.resolution,FID.getResolution())  # flatfile define resolution isnt always correct but whatever
        FID.getResolution() # u have to still call this otherwise ffsearch wont work and stuff

        # need to ensure loaded times are on same epoch, or do a conversion when plotting
        # loading files with same datastring names should either concatenate or just append a subnumber onto end

        self.loadData(ffTime, newDataStrings, datas, units)

        self.FIDs.append(FF_FD(PATH, FID))

        self.updateAfterOpening()
        return True

    def loadData(self, ffTime, newDataStrings, datas, units):
        # if all data strings currently exist in main data then append everything instead
        allMatch = True
        for dstr in newDataStrings:
            if dstr not in self.DATADICT:
                allMatch = False
                break

        if allMatch: # not sure if all dstrs need to necessarily match but simplest for now

            # since all our dstrs are present just get the current time of the first one
            arbStr = newDataStrings[0]

            curTime, curRes, curAvgRes = self.getTimes(arbStr,0)
            segments = Mth.getSegmentsFromTimeGaps(curRes, curAvgRes*2)
            f0 = ffTime[0]
            f1 = ffTime[-1]
            segLen = len(segments)
            for si in range(segLen): # figure out where this new file fits into current data based on time (before or after)
                s0 = segments[si][0]
                s1 = segments[si][1]
                t0 = curTime[s0]
                t1 = curTime[s1 - 1]

                startsBefore = f0 < t0 and f1 < t0  # starts entirely before this time
                startsAfter = f0 > t0 and f0 > t1   # starts entirely after this time
                startsBeforeFirst = startsBefore and si == 0
                startsAfterLast = startsAfter and si == segLen - 1
                if startsBefore or startsAfterLast:
                    if startsBeforeFirst:
                        joined = (ffTime, curTime)
                    elif startsAfterLast:
                        joined = (curTime, ffTime)
                    else:
                        joined = (curTime[:segments[si - 1][1] + 1], ffTime, curTime[s0:])

                    times = np.concatenate(joined)

                    resolutions = np.diff(times)
                    resolutions = np.append(resolutions, resolutions[-1]) # so same length as times
                
                    uniqueRes = np.unique(resolutions)
                    print(f'detected {len(uniqueRes)} resolutions')
                    #print(f'resolutions: {", ".join(map(str,uniqueRes))}')

                    # compute dif between each value and average resolution
                    avgRes = np.mean(resolutions)
                    self.resolution = min(self.resolution, avgRes)
                    self.TIMES[self.TIMEINDEX[arbStr]] = [times, resolutions, avgRes]

                    for di,dstr in enumerate(newDataStrings):
                        origData = self.ORIGDATADICT[dstr]

                        if startsBeforeFirst:
                            joinedData = (datas[di], origData)
                        elif startsAfterLast:
                            joinedData = (origData, datas[di])
                        else:
                            joinedData = (origData[:segments[si - 1][1] + 1], datas[di], origData[s0:])

                        self.ORIGDATADICT[dstr] = np.concatenate(joinedData)
                        self.DATADICT[dstr] = [Mth.interpolateErrors(self.ORIGDATADICT[dstr],self.errorFlag)]

                    print(f'CONCATENATING WITH EXISTING DATA')
                    break
                elif startsAfter:
                    continue
                else: # if flatfiles have overlapping time series then dont merge
                    print(f'ERROR: times overlap, no merge operation is defined for this yet')
                    return False

        else: # this is just standard new flatfile, cant be concatenated with current because it doesn't have matching column names

            self.DATASTRINGS.extend(newDataStrings)
            resolutions = np.diff(ffTime)
            resolutions = np.append(resolutions, resolutions[-1]) # append last value to make same length as time series
            uniqueRes = np.unique(resolutions)
            #print(uniqueRes)
            print(f'detected {len(uniqueRes)} resolutions')
            avgRes = np.mean(resolutions)

            self.resolution = min(self.resolution, avgRes)
            self.TIMES.append([ffTime, resolutions, avgRes])

            for i, dstr in enumerate(newDataStrings):
                self.TIMEINDEX[dstr] = len(self.TIMES) - 1 # index is of the time series we just added to end of list
                self.ORIGDATADICT[dstr] = datas[i]
                self.DATADICT[dstr] = [Mth.interpolateErrors(self.ORIGDATADICT[dstr],self.errorFlag)]
                self.UNITDICT[dstr] = units[i]

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


    def openCDF(self,PATH):#,q):
        """ opens a cdf file and loads the data into program structures """

        print(f'Opening CDF file: {PATH}')
        cdf = pycdf.CDF(PATH)
        if not cdf:
            print('CDF LOAD FAILED')
        self.cdfName = PATH

        #cdf data gets converted into this time epoch
        #this allows correct usage of FFTIME elsewhere in the code
        self.epoch = 'J2000'

        datas = []
        epochs = []
        epochLengths = set()
        for key,value in cdf.items():
            print(f'{key} : {value}')
            length = len(value)
            item = (key,length)
            if length <= 1:
                continue

            if 'epoch' in key.lower() or 'time' in key.lower():
                epochs.append(item)
                # not sure how to handle this scenario tho I think it would be rare if anything
                if length in epochLengths:
                    print(f'WARNING: 2 epochs with same length detected')
                epochLengths.add(length)
            else:
                datas.append(item)

        # pair epoch name with lists of data names that use that time
        epochsWithData = {}
        for en, el in epochs:
            edata = []
            for dn, dl in datas:
                if el == dl:
                    edata.append(dn)
            if len(edata) > 0:
                epochsWithData[en] = edata
            else:
                print(f'no data found for this epoch: {en}')

        # iterate through each column and try to convert and add to storage structures
        for key,dstrs in epochsWithData.items():
            print(f'{key} {len(cdf[key])}')

            startTime = time.time()
            print(f'converting time...')
            times = Mth.CDFEpochToTimeTicks(cdf[key][...])
            print(f'converted time in {time.time() - startTime} seconds')

            resolutions = np.diff(times)
            resolutions = np.append(resolutions,resolutions[-1])
            avgRes = np.mean(resolutions)
            self.resolution = min(self.resolution, avgRes)
            self.TIMES.append([times, resolutions, avgRes])

            for dstr in dstrs:
                data = cdf[dstr]
                zatrs = pycdf.zAttrList(data)
                shape = np.shape(data)
                fillVal = zatrs['FILLVAL']
                units = zatrs['UNITS']
                data = data[...]
                #print(zatrs)
                print(f'processing {dstr}, fillVal: {fillVal}, units: {units}')
                if len(shape) >= 2:
                    dim = shape[1]
                    # specific case with 3 or 4 component vectors
                    if len(shape) == 2 and (dim == 3 or dim == 4):
                        fix = ['X','Y','Z','T']
                        for i in range(dim):
                            newDstr = f'{dstr}_{fix[i]}'
                            self.DATASTRINGS.append(newDstr)
                            self.TIMEINDEX[newDstr] = len(self.TIMES) - 1
                            newData = data[:,i]
                            self.ORIGDATADICT[newDstr] = newData
                            self.DATADICT[newDstr] = [Mth.interpolateErrors(newData, fillVal)]
                            self.UNITDICT[newDstr] = units
                    else:
                        print(f'    skipping column: {dstr}, unhandled shape: {shape}')
                else:
                    self.DATASTRINGS.append(dstr)
                    self.TIMEINDEX[dstr] = len(self.TIMES) - 1
                    self.ORIGDATADICT[dstr] = data
                    self.DATADICT[dstr] = [Mth.interpolateErrors(data, fillVal)]
                    self.UNITDICT[dstr] = units

        self.calculateTimeVariables()
        self.calculateAbbreviatedDstrs()
        #print('\n'.join(self.ABBRV_DSTRS))

        self.enableToolsAndOptionsMenus(True)

        cdf.close()

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

        #for k,v in self.ABBRV_DSTR_DICT.items():
        #    print(f'{k} : {v}')

    
    def getMinAndMaxDateTime(self):
        minDateTime = UTCQDate.UTC2QDateTime(FFTIME(self.minTime, Epoch=self.epoch).UTC)
        maxDateTime = UTCQDate.UTC2QDateTime(FFTIME(self.maxTime, Epoch=self.epoch).UTC)
        return minDateTime,maxDateTime

    def getCurrentDateTime(self):
        return self.ui.timeEdit.start.dateTime(), self.ui.timeEdit.end.dateTime()

    def onStartSliderChanged(self, val):
        """callback for when the top (start) slider is moved"""
        self.iO = val

        # move tracker lines to show where new range will be
        tt = self.getTimeFromTick(self.iO)
        for line in self.trackerLines:
            line.show()
            line.setValue(tt-self.tickOffset)

        dt = UTCQDate.UTC2QDateTime(FFTIME(tt, Epoch=self.epoch).UTC)
        self.ui.timeEdit.setStartNoCallback(dt)

    def onEndSliderChanged(self, val):
        """callback for when the bottom (end) slider is moved"""
        self.iE = val

        # move tracker lines to show where new range will be
        tt = self.getTimeFromTick(self.iE)
        for line in self.trackerLines:
            line.show()
            line.setValue(tt - self.tickOffset + 1) #offset by linewidth so its not visible once released

        dt = UTCQDate.UTC2QDateTime(FFTIME(tt, Epoch=self.epoch).UTC)
        self.ui.timeEdit.setEndNoCallback(dt)

    def setSliderNoCallback(self, slider, i):
        ''' Update start/end slider with given value i '''
        # Determine which function to call to set value on
        if slider == 'start':
            f = self.ui.scrollSelect.set_start
        else:
            f = self.ui.scrollSelect.set_end

        # Block signals and set value
        self.ui.scrollSelect.blockSignals(True)
        f(i)
        self.ui.scrollSelect.blockSignals(False)

    def rangeChanged(self, rng):
        ''' Updates the view range with the current slider tick range '''
        # Hide tracker lines while updating range
        for line in self.trackerLines:
            line.hide()

        # Extract values
        start, stop = rng

        # Save start slider tick and update start time edit
        self.iO = start
        tt = self.getTimeFromTick(self.iO)
        dt = UTCQDate.UTC2QDateTime(FFTIME(tt, Epoch=self.epoch).UTC)
        self.ui.timeEdit.setStartNoCallback(dt)

        # Save stp[] slider tick and update start time edit
        self.iE = stop
        tt = self.getTimeFromTick(self.iE)
        dt = UTCQDate.UTC2QDateTime(FFTIME(tt, Epoch=self.epoch).UTC)
        self.ui.timeEdit.setEndNoCallback(dt)

        # Update view range by calling setTimes
        self.setTimes()

    def onStartEditChanged(self, val):
        """this gets called when the start date time edit is changed directly"""
        tick = FFTIME(UTCQDate.QDateTime2UTC(val), Epoch=self.epoch)._tick
        self.iO = self.calcTickIndexByTime(tick)
        self.setSliderNoCallback('start', self.iO)
        for line in self.trackerLines:
            line.hide()
        self.setTimes()

    def onEndEditChanged(self, val):
        """this gets called when the end date time edit is changed directly"""
        tick = FFTIME(UTCQDate.QDateTime2UTC(val), Epoch=self.epoch)._tick
        self.iE = self.calcTickIndexByTime(tick)
        self.setSliderNoCallback('stop', self.iE)
        for line in self.trackerLines:
            line.hide()
        self.setTimes()

    def setTimes(self):
        """function that updates both the x and y range of the plots based on current time variables"""

        # if giving exact same time index then slightly offset
        if self.iO == self.iE:
            if self.iE < self.iiE:
                self.iE += 1
            else:
                self.iO -= 1

        self.tO = self.getTimeFromTick(self.iO)
        self.tE = self.getTimeFromTick(self.iE)
        self.updateXRange()
        self.updateYRange()

    def getTimeFromTick(self, tick):
        assert(tick >= 0 and tick <= self.iiE)
        return self.minTime + (self.maxTime - self.minTime) * tick / self.iiE

    def updateXRange(self):
        for pi in self.plotItems:
            pi.setXRange(self.tO-self.tickOffset, self.tE-self.tickOffset, 0.0)
        self.pltGrd.setTimeLabel()

        # Update ticks/labels on bottom axis, clear top axis
        if self.pltGrd.labelSetGrd:
            startTick = self.tO-self.tickOffset
            endTick = self.tE-self.tickOffset
            self.pltGrd.labelSetGrd.setXRange(startTick, endTick, 0.0)

    # try to find good default plot strings
    def getDefaultPlotInfo(self):
        dstrs = []
        links = []
        keywords = ['BX','BY','BZ']
        if not self.insightMode:
            keywords.append('BT')
        for ki,kw in enumerate(keywords):
            row = []
            for dstr,abbrDstr in self.ABBRV_DSTR_DICT.items():
                allIn = True
                for c in kw:
                    if c.lower() not in abbrDstr.lower():
                        allIn = False
                        break
                if allIn:
                    row.append((dstr,0))
                    if self.insightMode:
                        break # only find one of each keyword
                    row.sort()
            if row:
                dstrs.append(row)
                links.append(len(dstrs)-1)

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
        if not links or len(dstrs) == 1:
            dstrs = [[(dstr, 0)] for dstr in self.DATASTRINGS[0:3]]
            links = [[i] for i in range(0, len(dstrs))]
        else:
            links = [links]

        return dstrs, links

    def plotDataDefault(self):
        dstrs, links = self.getDefaultPlotInfo()

        numPts = self.plotData(dstrs, links, [])

        # If a large number of points are plotted, enable downsampling for the plots
        if numPts > 2500000:
            self.ui.downsampleAction.setChecked(True)
            msg = "Plot data downsampled; disable under 'Options' Menu"
            self.ui.statusBar.showMessage(msg, 10000)
        else: # Otherwise, disable downsampling
            self.ui.downsampleAction.setChecked(False)

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

    def buildStackedLabel(self, plotStrings, colors):
        labels = []
        unitsSet = set()
        for dstr, en in plotStrings:
            lbl = self.getLabel(dstr, en)
            if lbl in self.ABBRV_DSTR_DICT:
                lbl = self.ABBRV_DSTR_DICT[lbl]
            labels.append(lbl)
            units = self.UNITDICT[dstr]
            unitsSet.add(units)

        unitsString = None
        if len(unitsSet) == 1:
            unitsString = unitsSet.pop()
            if unitsString == '':
                unitsString = None

        stackLbl = StackedLabel(labels, colors, units=unitsString)
        return stackLbl

    def onPlotRemoved(self, oldPlt):
        # Remove all linked region items from this plot and GeneralSelect
        # lists before removing the old plot
        for select in [self.currSelect, self.batchSelect, self.savedRegion]:
            if select is not None:
                select.onPlotRemoved(oldPlt)

    def addColorPltToGrid(self, plt, name, gradLbl, units=None):
        # Generate plot item and related elements from plot info
        labelTxt = gradLbl.getLabelText() if gradLbl is not None else None
        plotInfo = plt.getPlotInfo()
        newPlt, grad, gradLbl = self.loadColorPlot(plotInfo, labelTxt)

        # First, check if already plotted in main window
        pltIndex = self.pltGrd.getColorPlotIndex(name)
        if pltIndex is not None:
            self.pltGrd.replaceColorPlot(pltIndex, newPlt, name, grad, gradLbl, 
                units=units)
            self.plotItems[pltIndex] = newPlt
            self.onPlotRemoved(plt)

            # Update plot index used for selections
            vb = newPlt.getViewBox()
            vb.setPlotIndex(pltIndex)
        else:
            # Otherwise, add it to the end + update plotStrings list
            pltIndex = self.pltGrd.numPlots - 1
            self.pltGrd.addColorPlt(newPlt, name, grad, gradLbl, units=units)
            self.lastPlotStrings.append([(name, -1)])
            self.plotItems.append(newPlt)

            # Reset height factors
            self.lastPlotHeightFactors.append(1)
            self.pltGrd.setHeightFactors(self.lastPlotHeightFactors)

            # Also, update trace pens list if adding new plot
            self.plotTracePens.append(None)

        # Update local state information about color plot grid values
        self.colorPlotInfo[name] = (plotInfo, labelTxt, units)
        self.pltGrd.resizeEvent(None)
        self.pltGrd.adjustPlotWidths()

        # Return plot index to be used when creating link lists
        return pltIndex

    def loadColorPlot(self, plotInfo, labelTxt):
        # Extract specific plot info and offset times by main window's tick offset
        grid, x, y, logY, logColor, valRng, maskInfo = plotInfo
        x = x - self.tickOffset
        plotInfo = (grid, x, y, logY, logColor, valRng, maskInfo)

        # Generate plot item
        vb = MagPyViewBox(self, self.pltGrd.numPlots)
        plt = SimpleColorPlot(self.epoch, logY, vb=vb)
        plt.getAxis('bottom').tickOffset = self.tickOffset
        plt.loadPlotInfo(plotInfo)

        # Generate gradient legend and its axis label
        gradLegend = plt.getGradLegend(logColor)

        if labelTxt is not None:
            gradLbl = StackedAxisLabel(labelTxt)
            gradLbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))
        else:
            gradLbl = None

        return plt, gradLegend, gradLbl
    
    def addPlot(self, plt, label, pltStrs, links=None, hf=None, pens=None):
        # Close any selected tools
        self.closeFixSelection()
        self.closeBatchSelect()
        self.closeTraceStats()
        self.endGeneralSelect()

        # Add plot to grid
        self.pltGrd.addPlt(plt, label)

        # Add tracker line to plot item
        trackerLine = pg.InfiniteLine(movable=False, angle=90, pos=0, pen=self.trackerPen)
        plt.addItem(trackerLine)
        self.trackerLines.append(trackerLine)

        # Update selectable viewbox information
        vb = plt.getViewBox()
        vb._lastScene = None
        vb.window = self
        vb.plotIndex = len(self.plotItems)

        # Adjust datetime offsets
        for pdi in plt.listDataItems():
            times = pdi.xData
            pdi.setData(times-self.tickOffset, pdi.yData)
        plt.getAxis('bottom').tickOffset = self.tickOffset
        plt.getAxis('top').tickOffset = self.tickOffset

        # Adjust axes and buttons
        for ax in ['left', 'top', 'right', 'bottom']:
            plt.showAxis(ax)
        for ax in ['top', 'right']:
            plt.getAxis(ax).setStyle(showValues=False)
        plt.hideButtons()

        # Update state info for plotItems, plotStrings, plotLinks, 
        # heightFactors, and plotPens
        self.plotItems.append(plt)
        self.lastPlotStrings.append(pltStrs)

        if links is not None:
            self.lastPlotLinks.append(links)

        if hf is None:
            self.lastPlotHeightFactors.append(1)

        self.plotTracePens.append(pens)
        self.updateXRange()
        self.pltGrd.resizeEvent(None)

    def addSpectrogram(self, specData):
        ''' Adjusts specData before passing to pltGrd to load spectrogram '''
        # Adjust by tick offset
        specData.x_bins = specData.x_bins - self.tickOffset

        # Store specData in dictionary
        duplicate = specData.get_name() in self.SPECDICT
        self.SPECDICT[specData.get_name()] = specData

        # Replot if given spectrogram was previously plotted (in SPECDICT)
        if duplicate:
            self.plotAfterSpecUpdate()
            return

        # Add spectrogram to plotGrid and update state
        plt = self.pltGrd.addSpectrogram(specData)

        self.plotItems.append(plt)
        self.lastPlotStrings.append([(specData.get_name(), -1)])
        self.plotTracePens.append([None])

        self.updateXRange()

    def plotAfterSpecUpdate(self):
        ''' Replots data with same settings as before '''
        self.plotData(self.lastPlotStrings, self.lastPlotLinks, 
            self.lastPlotHeightFactors)

    def plotData(self, dataStrings, links, heightFactors):
        # Remove any saved linked regions from plots and save their state
        selectState = self.getSelectState()
        self.closeFixSelection()
        self.closeBatchSelect()
        self.disableTracker()

        # Clear any selected tools
        self.closeTraceStats()
        self.endGeneralSelect()

        # save what the last plotted strings and links are for other modules
        self.lastPlotStrings = dataStrings
        self.lastPlotLinks = links
        self.lastPlotHeightFactors = heightFactors

        self.plotItems = []
        self.labelItems = []

        # A list of pens for each trace (saved for consistency with spectra)
        self.plotTracePens = []

        # Store any previous label sets (for current file)
        prevLabelSets = []

        # Clear previous grid
        self.ui.glw.clear()

        # Add label for file name at top right
        fileList = [os.path.basename(FID.name) for FID in self.FIDs]
        fileNameLabel = FileLabel(fileList)
        self.fileNameLabel = fileNameLabel
        self.ui.glw.addItem(fileNameLabel, 0, 0, 1, 1)
        self.showFileLabel(self.ui.showFileLbl.isChecked())

        # Create new plot grid
        self.pltGrd = MainPlotGrid(self)
        self.pltGrd.setHeightFactors(heightFactors)
        self.ui.glw.addItem(self.pltGrd, 1, 0, 1, 1)

        self.trackerLines = []
        newColorPlotInfo = {} # Keep track of which color plots are re-plotted
        newSpec = {}

        numPts = 0
        for plotIndex, dstrs in enumerate(dataStrings):
            # Check if special plot
            colorPlt = False
            colorPltName = None
            for dstr, en in dstrs:
                if en < 0:
                    colorPlt = True
                    colorPltName = dstr

            if colorPlt:
                # Save old color plot info and generate spectrogram
                specData = self.SPECDICT[colorPltName]
                newSpec[colorPltName] = specData
                plt = self.pltGrd.addSpectrogram(specData)
                self.plotItems.append(plt)
                self.plotTracePens.append([None])
                continue

            vb = MagPyViewBox(self, plotIndex)
            pi = MagPyPlotItem(epoch=self.epoch, viewBox=vb)
            vb.enableAutoRange(x=False, y=False) # range is being set manually in both directions

            pi.ctrl.logYCheck.toggled.connect(functools.partial(self.updateLogScaling, plotIndex))

            # add some lines used to show where time series sliders will zoom to
            trackerLine = pg.InfiniteLine(movable=False, angle=90, pos=0, pen=self.trackerPen)
            pi.addItem(trackerLine)
            self.trackerLines.append(trackerLine)

            # add horizontal zero line
            zeroLine = pg.InfiniteLine(movable=False, angle=0, pos=0)
            zeroLine.setPen(pg.mkPen('#000000', width=1, style=QtCore.Qt.DotLine))
            pi.addItem(zeroLine, ignoreBounds=True)

            pi.hideButtons() # hide autoscale button

            # show top and right axis, but hide labels (they are off by default apparently)
            la = pi.getAxis('left')
            la.style['textFillLimits'] = [(0,1.1)] # no limits basically to force labels by each tick no matter what
            #la.setWidth(50) # this also kinda works but a little space wasteful, saving as reminder incase dynamic solution messes up

            ba = pi.getAxis('bottom')
            ba.style['textFillLimits'] = [(2, 0.45)]
            ta = pi.getAxis('top')
            ra = pi.getAxis('right')
            ta.show()
            ra.show()
            ta.setStyle(showValues=False)
            ra.setStyle(showValues=False)

            # only show tick labels on bottom most axis
            if plotIndex != len(dataStrings) - 1:
                ba.setStyle(showValues=False)

            tracePens = []
            dstrList = []
            colorsList = []

            self.plotItems.append(pi) #save it for ref elsewhere

            # add traces on this plot for each dstr
            for i, (dstr, editNum) in enumerate(dstrs):
                if dstr == '':
                    continue

                # figure out which pen to use
                numPens = len(self.pens)
                if len(dstrs) == 1: # if just one trace then base it off which plot
                    penIndex = plotIndex % numPens
                    pen = self.pens[penIndex]
                elif i >= numPens:
                    # If past current number of pens, generate a random one and
                    # add it to the standard pen list for consistency between plots
                    pen = self.genRandomPen()
                    self.pens.append(pen)
                else: # else if base off trace index, capped at pen count
                    penIndex = min(i,numPens - 1)
                    pen = self.pens[penIndex]

                # If user set custom trace pen through plotAppr, use that pen instead,
                # searching through the customPens list for a match
                if plotIndex < len(self.customPens):
                    if i < len(self.customPens[plotIndex]):
                        prevDstr, prevEn, prevPen = self.customPens[plotIndex][i]
                        if prevDstr == dstr and prevEn == editNum:
                            pen = prevPen

                #save pens so spectra can stay synced with main plot
                tracePens.append(pen)

                pts = self.plotTrace(pi, dstr, editNum, pen)
                numPts += pts
                dstrList.append(dstr)
                colorsList.append(pen.color().name())

            self.plotTracePens.append(tracePens)

            # set plot to current range based on time sliders
            pi.setXRange(self.tO-self.tickOffset, self.tE-self.tickOffset, 0.0)

            #todo: needs to be set for current min and max of all time ranges
            pi.setLimits(xMin=self.minTime-self.tickOffset, xMax=self.maxTime-self.tickOffset)

            # Create plot label and add to grid
            stckLbl = self.buildStackedLabel(dstrs, colorsList)
            self.pltGrd.addPlt(pi, stckLbl)

        ## end of main for loop

        # Downsample data if checked
        self.enableDownsampling(self.ui.downsampleAction.isChecked())

        # Add in all previous label sets, if there are any
        for labelSetDstr in prevLabelSets:
            self.pltGrd.addLabelSet(labelSetDstr)

        self.updateXRange()
        self.updateYRange()

        for pi in self.plotItems:
            for item in pi.items:
                item.viewRangeChanged()
            
        self.colorPlotInfo = newColorPlotInfo

        # Rebuild any saved selections
        self.loadSelectState(selectState)

        # Return the total number of points plotted through plotTrace function
        self.pltGrd.resizeEvent(None)

        # Enable tracker if option checked
        if self.ui.toggleTrackerAction.isChecked():
            self.enableTracker()

        self.SPECDICT = newSpec
        return numPts
    
    def enableTracker(self):
        ''' Creates a new tracker line item '''
        if not self.hoverTracker:
            self.hoverTracker = TrackerRegion(self, self.plotItems, mode='Single', color='#383838')
            self.hoverTracker.hideLabel()

    def disableTracker(self):
        ''' Removes any tracker lines present in plot grid '''
        if self.hoverTracker:
            self.hoverTracker.setAllRegionsVisible(False)
            self.hoverTracker.deleteLater()
            self.hoverTracker = None
            self.ui.timeStatus.setText('')
    
    def toggleTracker(self):
        ''' Toggles tracker line '''
        if self.hoverTracker:
            self.disableTracker()
        else:
            self.enableTracker()

    def enableScrolling(self, val):
        # Minimum plot height set to 3 inches for now
        min_height = self.minimumPlotHeight * QtGui.QDesktopWidget().logicalDpiY()

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
        viewHeightInches = (winHeight / QtGui.QDesktopWidget().logicalDpiY()) - 2
        minBound = max(viewHeightInches/len(self.plotItems), 1)

        # Decreases the minimum plot height used to set gview size
        # when scrolling is enabled
        if self.ui.enableScrollingAction.isChecked():
            self.minimumPlotHeight = max(self.minimumPlotHeight - 0.5, minBound)
            self.enableScrolling(True)

    def enableDownsampling(self, val):
        if val:
            for plt in self.plotItems:
                plt.setDownsampling(ds=None, auto=True, mode='peak')
                plt.setClipToView(True)
        else:
            for plt in self.plotItems:
                plt.setDownsampling(ds=False)
                plt.setClipToView(False)

    def genRandomPen(self):
        r = np.random.randint(low=0, high=255)
        g = np.random.randint(low=0, high=255)
        b = np.random.randint(low=0, high=255)
        return pg.mkPen([r,g,b])

    def replotDataCallback(self):
        # done this way to ignore the additional information ui callbacks will provide
        self.replotData()
        
    def replotData(self, desiredEdit=None):
        """simply redraws the traces and ensures y range is correct without rebuilding everything
           if desiredEdit is defined it will try to plot strings at that edit if they have data there
        """

        newPltStrs = []
        for i in range(len(self.plotItems)):
            pi = self.plotItems[i]
            plotStrs = self.lastPlotStrings[i]
            pens = self.plotTracePens[i]
            if pens == [None]:
                continue
            if self.pltGrd.colorPltKws[i] is not None:
                newPltStrs.append(plotStrs)
                continue
            pi.clearPlots()

            # keep track of the frequency of strings in each plot (regardless of edit number)
            if desiredEdit is not None:
                seenCount = {} # how many times this string has been seen
                for dstr,en in plotStrs:
                    if dstr in seenCount:
                        seenCount[dstr]+=1
                    else:
                        seenCount[dstr] = 1

            subPltStrs = []
            j = 0
            while j < len(plotStrs):
                dstr, editNum = plotStrs[j]
                prevEdit = editNum
                if editNum < 0:
                    subPltStrs.append((dstr, editNum))
                    j += 1
                    continue

                edits = self.DATADICT[dstr]

                # if u have multiple edits of same data string on this plot then ignore the desiredEdit option
                if desiredEdit is not None and seenCount[dstr] == 1:
                    # if string has data with this edit
                    if len(edits[desiredEdit]) > 0:
                        editNum = desiredEdit
                elif editNum >= len(edits): # if bad edit number then delete (happens usually when u delete an edit that is currently plotted)
                    del plotStrs[j]
                    continue

                plotStrs[j] = dstr,editNum #save incase changes were made (so this reflects elsewhere)

                self.plotTrace(pi, dstr, editNum, pens[j])

                # Update custom pens
                if i < len(self.customPens):
                    cstmPenIndex = 0
                    for cstmDstr, cstmEn, cstmPen in self.customPens[i]:
                        if cstmDstr == dstr and cstmEn == prevEdit:
                            self.customPens[i][cstmPenIndex] = (dstr, editNum, cstmPen)
                        cstmPenIndex += 1

                subPltStrs.append((dstr, editNum))

                j+=1
            newPltStrs.append(subPltStrs)
        self.lastPlotStrings = newPltStrs
        self.updateYRange()

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

    def getConnectionList(self, resolutions, avgRes):
        mask = resolutions > (avgRes * 2)
        segments = np.array(np.logical_not(mask), dtype=np.int32)
        return segments

    # both plotData and replot use this function internally
    def plotTrace(self, pi, dstr, editNumber, pen):
        Y = self.getData(dstr, editNumber)
        if len(Y) <= 1: # not sure if this can happen but just incase
            print(f'Error: insufficient Y data for column "{dstr}"')
            return 0
        errMask = abs(Y) < abs(self.errorFlag)

        times,resolutions,avgRes = self.getTimes(dstr, editNumber)
        Y = Y[errMask]
        times = times[errMask]
        resolutions = np.diff(times)
        resolutions = np.concatenate([resolutions, [resolutions[-1]]])

        # Find smallest tick in data
        ofst = self.minTime
        self.tickOffset = ofst
        pi.tickOffset = ofst
        pi.getAxis('bottom').tickOffset = ofst
        pi.getAxis('top').tickOffset = ofst
        # Subtract offset value from all times values for plotting
        ofstTimes = times - ofst

        # Determine data segments/type and plot
        if not self.ui.bridgeDataGaps.isChecked():
            # Find segments of data that should not be connected due to time gaps
            segs = self.getConnectionList(resolutions, avgRes)
            if self.ui.drawPoints.isChecked():
                brush = pg.mkBrush(pen.color())
                outlinePen = self.getPointsOutlinePen(pen)
                pi.scatterPlot(ofstTimes, Y, pen=outlinePen, brush=brush, size=2, 
                    connect=segs, name=dstr)
            else:
                pdi = MagPyPlotDataItem(ofstTimes, Y, pen=pen, connect=segs, 
                    name=dstr)
                pi.addItem(pdi)
        else:
            if self.ui.drawPoints.isChecked():
                brush = pg.mkBrush(pen.color())
                outlinePen = self.getPointsOutlinePen(pen)
                pi.scatterPlot(ofstTimes, Y, pen=outlinePen, brush=brush, size=2,
                    name=dstr)
            else:
                pdi = MagPyPlotDataItem(ofstTimes, Y, pen=pen, name=dstr)
                pi.addItem(pdi)

        return len(Y)

    def getPointsOutlinePen(self, pen):
        color = pen.color()
        r, g, b, alpha = color.getRgb()
        new_r = max(r-25, 0)
        new_g = max(g-25, 0)
        new_b = max(b-25, 0)
        return pg.mkPen(new_r, new_g, new_b)

    def updateYRange(self):
        """
        this function scales Y axis to have equally scaled ranges but not the same actual range
        pyqtgraph has built in y axis linking but doesn't work exactly how we want
        also this replicates pyqtgraph setAutoVisible to have scaling for currently selected time vs the whole file
        """
        if self.lastPlotStrings is None or len(self.lastPlotStrings) == 0:
            return
        values = [] # (min,max)
        skipRangeSet = set() # set of plots where the min and max values are infinite so they should be ignored
        # for each plot, find min and max values for current time selection (consider every trace)
        # otherwise just use the whole visible range * -> self.iiE
        outOfRangeCount = 0
        for plotIndex, dstrs in enumerate(self.lastPlotStrings):
            minVal = np.inf
            maxVal = -np.inf

            logPlot = self.plotItems[plotIndex].ctrl.logYCheck.isChecked()

            # find min and max values out of all traces on this plot
            for (dstr,editNum) in dstrs:
                if editNum < 0:
                    skipRangeSet.add(plotIndex)
                    continue
                Y = self.getData(dstr,editNum)

                if logPlot: # Use *valid* log values to get range
                    Y = np.log10(Y[Y>0])

                if self.ui.scaleYToCurrentTimeAction.isChecked():
                    X = self.getTimes(dstr,editNum)[0] # first in list is time series
                    a = self.calcDataIndexByTime(X, self.tO)
                    b = self.calcDataIndexByTime(X, self.tE)
                    if a == b: # both are out of range on same side so data shouldnt be plotted
                        #print(f'"{dstr}" out of range, not plotting') # can get rid of this warning later but want to see it in action first
                        outOfRangeCount += 1
                        continue

                    if a > b: # so sliders work either way
                        a,b = b,a

                    Y = Y[a:b] # get correct slice

                Y = Y[Y<self.errorFlag]

                minVal = min(minVal, np.min(Y))
                maxVal = max(maxVal, np.max(Y))
            # if range is bad then dont change this plot
            if np.isnan(minVal) or np.isinf(minVal) or np.isnan(maxVal) or np.isinf(maxVal):
                skipRangeSet.add(plotIndex)
            values.append((minVal,maxVal))

        if outOfRangeCount > 0:
            print(f'{outOfRangeCount} data columns out of range, not plotting')

        for row in self.lastPlotLinks:
            # find largest range in group
            largest = 0
            for i in row:
                if i in skipRangeSet or i >= len(values):
                    continue
                diff = values[i][1] - values[i][0]
                largest = max(largest,diff)

            # then scale each plot in this row to the range
            for i in row:
                if i in skipRangeSet or i >= len(values):
                    continue
                diff = values[i][1] - values[i][0]
                l2 = (largest - diff) / 2.0
                self.plotItems[i].setYRange(values[i][0] - l2, values[i][1] + l2, padding = 0.05)

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
            resolutions = np.diff(times)
            avgRes = np.median(resolutions)
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
        # Get original file ID
        FID = self.FIDs[0]

        # Get information about original file
        filename = FID.getName()
        asc_imp = Asc_Importer(self, filename)
        header = asc_imp.header
        asc_type = FID.fileType
        epoch = FID.ancInfo['Epoch']
        timeMode = FID.ancInfo['TimeMode']
        cols = None
        if asc_type == 'Fixed Columns':
            cols = asc_imp.guessColumns(header, asc_imp.lines[1])

        # Build records
        records, labels = self.getLatestRecords()

        # Pass the old file format to ASC_Output and generate the new file
        asc_out = ASC_Output(name, records, header, (timeMode == 'Seconds'), asc_type, cols, epoch)
        asc_out.write()

    def exportFlatFileCopy(self, name):
        ''' Create a new flat file based on the loaded flat file 
            using the latest data
        '''
        # Build records
        records, labels = self.getLatestRecords()

        # Get units, sources, and epoch information from original flat file
        units = [self.UNITDICT[label] for label in labels]
        sources = ['']*len(labels)
        epoch = self.epoch

        # Generate flat file
        createFF(name, records[:,0], records[:,1:], labels, units, sources, epoch)

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

    def mouseDragPlots(self, ev):
        # Allow mouse dragging only in stats mode
        if self.currSelect is not None and self.currSelect.name != 'Stats':
            return

        # Check movement direction so plot widths do not shrink when
        # dragging past edges
        lastPos = ev.lastPos()
        currPos = ev.pos()
        leftMove = True if lastPos.x() - currPos.x() < 0 else False
        rightMove = True if lastPos.x() - currPos.x() > 0 else False
        if self.tE >= self.maxTime and rightMove:
            return
        elif self.tO <= self.minTime and leftMove:
            return

        # Signal mouse drag event to all plots' viewboxes
        for plt in self.plotItems:
            vb = plt.getViewBox()
            vb.setMouseEnabled(x=True,y=False)
            vb.blockSignals(True)
            pg.ViewBox.mouseDragEvent(vb, ev)
            vb.blockSignals(False)

        # Get new x value range and update time edits / sliders (indirectly)
        xMin, xMax = vb.state['targetRange'][0]
        self.tO = xMin + self.tickOffset
        self.tE = xMax + self.tickOffset
        minDate = self.getDateTimeFromTick(self.tO)
        maxDate = self.getDateTimeFromTick(self.tE)
        self.ui.timeEdit.start.setDateTime(minDate)
        self.ui.timeEdit.end.setDateTime(maxDate)

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
        i0 = self.calcTickIndexByTime(FFTIME(UTCQDate.QDateTime2UTC(timeEdit.start.dateTime()), Epoch=self.epoch)._tick)
        i1 = self.calcTickIndexByTime(FFTIME(UTCQDate.QDateTime2UTC(timeEdit.end.dateTime()), Epoch=self.epoch)._tick)
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
        for pltNum in range(0, len(region.regionItems)):
            if region.isVisible(pltNum) and not self.plotItems[pltNum].isSpecialPlot():
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
            self.startTraceStats()
        self.currSelect.addRegion(t0, t1)

    def getCurrentTool(self, setTrace=True):
        if self.currSelect:
            return self.currSelect
        elif setTrace:
            self.startTraceStats()
            return self.currSelect
        else:
            return None

    def gridLeftClick(self, x, plotIndex, ctrlPressed):
        # Get current tool selection (or set trace as select tool if batch select
        # is not open) and pass the left click to it
        batchOpen = self.batchSelect is not None
        tool = self.getCurrentTool(setTrace=(not batchOpen))
        if tool:
            tool.leftClick(x, plotIndex, ctrlPressed)

        # If batch select is open and the selections are not locked, then
        # pass the left click to it as well
        if batchOpen and not self.batchSelect.isLocked():
            tool = self.batchSelect.linkedRegion
            tool.leftClick(x, plotIndex, ctrlPressed)

    def gridRightClick(self, plotIndex):
        # Apply right click to current tool
        tool = self.getCurrentTool(setTrace=False) # Don't set trace as select
        if tool:
            tool.rightClick(plotIndex)

        # Return whether this click was applied to a selection or not so the viewbox
        # can determine whether to apply the default right click action instead
        res = True if tool is not None else False
        return res

    def hoverStart(self, pos):
        ''' Action when mouse hover started '''
        # Ignore if hover tracker object hasn't been created
        if not self.hoverTracker:
            return

        # Set hover tracker visible
        for i in range(0, len(self.plotItems)):
            self.hoverTracker.setVisible(True, i)
        
        # Set tracker position
        self.gridHover(pos)

    def gridHover(self, pos):
        ''' Action as mouse hover continues '''
        # Ignore if hover tracker object hasn't been created
        if not self.hoverTracker:
            return

        # Adjust tick position by offset and set tracker position
        x = pos.x() + self.tickOffset
        self.hoverTracker.setRegion((x-self.tickOffset, x-self.tickOffset))

        # Update timestamp in statusbar
        ts = FFTIME(x, Epoch=self.epoch).UTC
        self.ui.timeStatus.setText(ts+' ')

    def hoverEnd(self):
        ''' Action when hover ends '''
        if not self.hoverTracker:
            return
        
        # Set hover tracker hidden
        for i in range(0, len(self.plotItems)):
            self.hoverTracker.setVisible(False, i)
        
        # Clear timestamp in statusbar
        self.ui.timeStatus.setText('')

# look at the source here to see what functions you might want to override or call
#http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ViewBox/ViewBox.html#ViewBox
class MagPyViewBox(SelectableViewBox): # custom viewbox event handling
    def __init__(self, window, plotIndex, *args, **kwds):
        SelectableViewBox.__init__(self, window, plotIndex, *args, **kwds)
        self.window = window
        self.plotIndex = plotIndex
        self.menuSetup()

    def menuSetup(self):
        # Remove menu options that won't be used
        actions = self.menu.actions()
        xAction, yAction, mouseAction = actions[1:4]
        for a in [xAction, mouseAction]:
            self.menu.removeAction(a)
    
    def setPlotIndex(self, index):
        self.plotIndex = index
    
    def hoverEvent(self, ev):
        # Update hover tracker position if present
        if ev.isEnter():
            # Start hover
            pos = self.mapToView(ev.pos())
            self.window.hoverStart(pos)
        elif ev.isExit():
            # End hover
            self.window.hoverEnd()
        else:
            # Start and update hover
            pos = self.mapToView(ev.pos())
            self.window.hoverStart(pos)
            self.window.gridHover(pos)
        
# Wrapper class for Flat File FID functions
class FF_FD():
    def __init__(self, filename, FID):
        self.FID = FID
        self.name = filename

    def getFileType(self):
        return 'FLAT FILE'

    def getName(self):
        return self.name

    def getEpoch(self):
        return self.FID.getEpoch()

    def getUnits(self):
        return self.FID.getColumnDescriptor('UNITS')

    def getLabels(self):
        return self.FID.getColumnDescriptor('NAME')

    def getRows(self):
        return self.FID.getRows()

    def ffSearch(self, tick, startRow, endRow):
        return self.FID.ffsearch(tick, startRow, endRow)

    def getRecords(self):
        nRows = self.getRows()
        records = self.FID.DID.sliceArray(row=1, nRow=nRows)
        return records["time"], records["data"]

    def open(self):
        self.FID.open()

    def close(self):
        self.FID.close()

def myexepthook(type, value, tb):
    print(f'{type} {value}')
    traceback.print_tb(tb,limit=5)
    os.system('pause')

def startMagPy(ffLst=None):
    '''
    Main function for creating MagPy4Window object and starting program
    '''
    # Set up application
    app = QtWidgets.QApplication(sys.argv)

    app.setOrganizationName('IGPP UCLA')
    app.setOrganizationDomain('igpp.ucla.edu')
    app.setApplicationName('MagPy4')

    # Set fusion as default style if found
    keys = QtGui.QStyleFactory().keys()
    if 'Fusion' in keys:
        app.setStyle(QtGui.QStyleFactory.create('Fusion'))

    # Create the MagPy4 window
    main = MagPy4Window(app)
    main.showMaximized()

    # Initialize any files passed
    if ffLst is not None:
        main.openFileList(ffLst, True, True)

    args = sys.argv
    sys.excepthook = myexepthook
    sys.exit(app.exec_())

def runMarsPy():
    runMagPy()

def runMagPy():
    # Read in arguments, opening MagPy if the update flag was not passed,
    # and passing along any the names of any files to open at startup
    res, ffLst = readArgs()
    if res:
        startMagPy(ffLst=ffLst)

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
