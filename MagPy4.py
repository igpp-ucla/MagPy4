"""
Main module for the program
handles data, plotting and main window management
"""

# python 3.6
import os
import sys

# so python looks in paths for these folders too
# maybe make this into actual modules in future
sys.path.insert(0, 'ffPy')
sys.path.insert(0, 'cdfPy')

# Version number and copyright notice displayed in the About box
NAME = f'MagPy4'
VERSION = f'Version 1.0.1.0 (February 13, 2019)'
COPYRIGHT = f'Copyright © 2019 The Regents of the University of California'

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
import pyqtgraph as pg

import FF_File
from FF_Time import FFTIME, leapFile

from MagPy4UI import MagPy4UI, PyQtUtils, PlotGrid, StackedLabel, TimeEdit
from plotMenu import PlotMenu
from spectra import Spectra
from dataDisplay import DataDisplay, UTCQDate
from plotAppearance import MagPyPlotApp
from addTickLabels import AddTickLabels
from edit import Edit
from traceStats import TraceStats
from helpWindow import HelpWindow
from AboutDialog import AboutDialog
from pyqtgraphExtensions import DateAxis, LinkedAxis, PlotPointsItem, PlotDataItemBDS, BLabelItem, LinkedRegion
from MMSTools import PlaneNormal, Curlometer
from mth import Mth
from tests import Tests
import bisect

import time
import functools
import multiprocessing as mp
import traceback

CANREADCDFS = False
try:
    import pycdf
    CANREADCDFS = True
except Exception as e:
    print(f'ERROR: CDF loading disabled!')
    print(f'check README for instructions on how to install CDF C Library')
    print(e)
    print(' ')

class MagPy4Window(QtWidgets.QMainWindow, MagPy4UI):
    def __init__(self, app, parent=None):
        super(MagPy4Window, self).__init__(parent)

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

        self.ui.startSlider.valueChanged.connect(self.onStartSliderChanged)
        self.ui.endSlider.valueChanged.connect(self.onEndSliderChanged)
        self.ui.startSlider.sliderReleased.connect(self.setTimes)
        self.ui.endSlider.sliderReleased.connect(self.setTimes)

        # Shift window connections
        self.ui.mvRgtBtn.clicked.connect(self.shiftWinRgt)
        self.ui.mvLftBtn.clicked.connect(self.shiftWinLft)
        self.ui.mvLftShrtct.activated.connect(self.shiftWinLft)
        self.ui.mvRgtShrtct.activated.connect(self.shiftWinRgt)
        self.ui.shftPrcntBox.valueChanged.connect(self.updtShftPrcnt)

        self.ui.timeEdit.start.dateTimeChanged.connect(self.onStartEditChanged)
        self.ui.timeEdit.end.dateTimeChanged.connect(self.onEndEditChanged)

        # Main menu action connections
        self.ui.actionOpenFF.triggered.connect(functools.partial(self.openFileDialog, True,True))
        self.ui.actionAddFF.triggered.connect(functools.partial(self.openFileDialog, True, False))
        self.ui.actionOpenCDF.triggered.connect(functools.partial(self.openFileDialog,False,True))
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionShowData.triggered.connect(self.showData)
        self.ui.actionPlotMenu.triggered.connect(self.openPlotMenu)
        self.ui.actionSpectra.triggered.connect(self.startSpectra)
        self.ui.actionEdit.triggered.connect(self.openEdit)
        self.ui.actionHelp.triggered.connect(self.openHelp)
        self.ui.actionAbout.triggered.connect(self.openAbout)
        self.ui.switchMode.triggered.connect(self.swapMode)
        self.ui.runTests.triggered.connect(self.runTests)

        self.ui.actionPlaneNormal.triggered.connect(self.openPlaneNormal)
        self.ui.actionCurlometer.triggered.connect(self.openCurlometer)

        # Content menu action connections
        self.ui.plotApprAction.triggered.connect(self.openPlotAppr)
        self.ui.addTickLblsAction.triggered.connect(self.openAddTickLbls)

        # options menu dropdown
        self.ui.scaleYToCurrentTimeAction.triggered.connect(self.updateYRange)
        self.ui.antialiasAction.triggered.connect(self.toggleAntialiasing)
        self.ui.bridgeDataGaps.triggered.connect(self.replotDataCallback)
        self.ui.drawPoints.triggered.connect(self.replotDataCallback)

        # Disable the Tools and Options menus. They'll be enabled after the user opens a file.
        self.enableToolsAndOptionsMenus(False)

        self.dataDisplay = None
        self.plotMenu = None
        self.spectra = None
        self.plotAppr = None
        self.addTickLbls = None
        self.edit = None
        self.traceStats = None
        self.helpWindow = None
        self.aboutDialog = None
        self.FIDs = []
        self.tickOffset = 0 # Smallest tick in data, used when plotting x data

        # MMS Tools
        self.planeNormal = None
        self.curlometer = None

        # these are saves for options for program lifetime
        self.plotMenuCheckBoxMode = False
        self.traceStatsOnTop = True

        self.initDataStorageStructures()

        # this is where options for plot lifetime are saved
        self.initVariables()

        # Shift percentage setup
        self.shftPrcnt = self.ui.shftPrcntBox.value()/100 # Initialize default val

        # Cutoff values for time-label properties
        self.dayCutoff = 60 * 60 * 24
        self.hrCutoff = 60 * 60
        self.minCutoff = 10 * 60

        self.magpyIcon = QtGui.QIcon()
        self.marsIcon = QtGui.QIcon()
        if self.OS == 'mac':
            self.magpyIcon.addFile('images/magPy_blue.hqx')
            self.marsIcon.addFile('images/mars.hqx')
        else:
            self.magpyIcon.addFile('images/magPy_blue.ico')
            self.marsIcon.addFile('images/mars.ico')

        self.app.setWindowIcon(self.magpyIcon)

        # setup pens
        self.pens = []
        # Blue, Green, Red, Yellow, Magenta, Black
        colors = ['#0000ff','#00ad05','#ea0023','#fc9f00', '#ce0d9e', '#000000']
        for c in colors:
            self.pens.append(pg.mkPen(c, width=1))# style=QtCore.Qt.DotLine)
        self.trackerPen = pg.mkPen('#000000', width=1, style=QtCore.Qt.DashLine)
        self.customPens = []

        self.pltGrd = None
        self.plotItems = []
        self.trackerLines = []
        self.regions = []

        self.startUp = True

    def shiftWindow(self, direction):
        winWidth = abs(self.iE - self.iO) # Number of ticks currently displayed
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

        # Update slider values and self.iO, self.iE
        self.setSliderNoCallback(self.ui.startSlider, newTO)
        self.setSliderNoCallback(self.ui.endSlider, newTE)

        # Update timeEdit values
        self.onStartSliderChanged(newTO)
        self.onEndSliderChanged(newTE)
        self.ui.timeEdit.start.update() # Update appearance for OSX users
        self.ui.timeEdit.end.update()

        # Update plots
        self.setTimes()

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

    # Use these two functions to set a temporary status msg and clear it
    def showStatusMsg(self, msg):
        status = 'STATUS: ' + msg
        self.ui.statusBar.showMessage(status)

    def clearStatusMsg(self):
        self.ui.statusBar.clearMessage()

    def enableToolsAndOptionsMenus(self, bool):
        """Enable or disable the Tools and Options menus.
        """
        self.ui.toolsMenu.setEnabled(bool)
        self.ui.optionsMenu.setEnabled(bool)

    # close any subwindows if main window is closed
    # this should also get called if flatfile changes
    def closeEvent(self, event):
        self.closeAllSubWindows()

    def closeAllSubWindows(self):
        self.closePlotMenu()
        self.closePlotAppr()
        self.closeEdit()
        self.closeData()
        self.closeTraceStats()
        self.closeSpectra()
        self.closeAddTickLbls()
        self.closePlaneNormal()
        self.closeCurlometer()

    def initVariables(self):
        """init variables here that should be reset when file changes"""
        self.lastPlotStrings = None
        self.lastPlotLinks = None
        self.selectMode = None
        self.currentEdit = 0 # current edit number selected
        self.editNames = [] # list of edit names, index into list is edit number
        self.editHistory = []
        self.customPens = []
        self.pltGrd = None
        self.regions = []
        
    def closePlotMenu(self):
        if self.plotMenu:
            self.plotMenu.close()
            self.plotMenu = None

    def closeEdit(self):
        if self.edit:
            self.edit.close()
            self.edit = None

    def closeData(self):
        if self.dataDisplay:
            self.dataDisplay.close()
            self.dataDisplay = None

    def closeTraceStats(self):
        if self.traceStats:
            self.traceStats.close()
            self.traceStats = None

    def closeSpectra(self):
        if self.spectra:
            self.spectra.close()
            self.spectra = None

    def closeHelp(self):
        if self.helpWindow:
            self.helpWindow.close()
            self.helpWindow = None

    def closeAbout(self):
        if self.aboutDialog:
            self.aboutDialog.close()
            self.aboutDialog = None

    #thoughts on refactor this into using a dictionary, so youd call close with string arg of window name??
    #def closeSubWindow(key):
    #    if key not in self.subWindows:
    #        print('prob bad')
    #        return
    #    if not self.subWindows[key]:
    #        return
    #    self.subWindows[key].close()
    #    self.subWindows[key] = None

    def openCurlometer(self):
        self.closeCurlometer()
        self.curlometer = Curlometer(self)
        self.initGeneralSelect('Curlometer', '#ffa500', self.curlometer.ui.timeEdit)

    def showCurlometer(self):
        if self.curlometer:
            self.curlometer.show()
            self.curlometer.calculate()

    def closeCurlometer(self):
        if self.curlometer:
            self.curlometer.close()
            self.curlometer = None

    def openPlaneNormal(self):
        self.closePlaneNormal()
        self.planeNormal = PlaneNormal(self)
        self.initGeneralSelect('Plane Normal', '#42f495', None)

    def showNormal(self):
        if self.planeNormal:
            self.planeNormal.show()
            self.planeNormal.calculate()

    def closePlaneNormal(self):
        if self.planeNormal:
            self.planeNormal.close()
            self.planeNormal = None
            self.endGeneralSelect()

    def openPlotMenu(self):
        self.closePlotMenu()
        self.plotMenu = PlotMenu(self)

        geo = self.geometry()
        self.plotMenu.move(geo.x() + 200, geo.y() + 100)
        self.plotMenu.show()

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
        self.edit = Edit(self)
        self.edit.show()
        
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
        self.dataDisplay = DataDisplay(self, self.FIDs, Title='Flatfile Data')
        self.dataDisplay.show()

    def openTraceStats(self, plotIndex):
        self.closeSpectra()
        self.traceStats = TraceStats(self, plotIndex)
        self.traceStats.show()

    def startSpectra(self):
        self.closeTraceStats()
        if not self.spectra or self.spectra.wasClosed:
            self.spectra = Spectra(self)
            self.showStatusMsg('Selecting spectra range...')
            self.initGeneralSelect('Spectra', '#c551ff', self.spectra.ui.timeEdit, True)

    def showSpectra(self):
        if self.spectra:
            self.clearStatusMsg()
            self.spectra.show()
            self.spectra.initPlots()
            PyQtUtils.moveToFront(self.spectra)

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
        if self.spectra:
            self.spectra.updateSpectra()

    def getPrunedData(self, dstr, en, a, b):
        """returns data with error values removed and nothing put in their place (so size reduces)"""
        data = self.getData(dstr, en)[a:b]
        return data[data < self.errorFlag]

    def swapMode(self): #todo: add option to just compile to one version or other with a bool swap as well
        """swaps default settings between marspy and magpy"""
        txt = self.ui.switchMode.text()
        self.insightMode = not self.insightMode
        txt = 'Switch to MMS' if self.insightMode else 'Switch to MarsPy'
        # Hide or show MMS tools menu
        self.ui.showMMSMenu(not self.insightMode)
        tooltip = 'Loads various presets specific to the MMS mission and better for general use cases' if self.insightMode else 'Loads various presets specific to the Insight mission'
        self.ui.switchMode.setText(txt)
        self.ui.switchMode.setToolTip(tooltip)
        self.plotDataDefault()
        self.setWindowTitle('MarsPy' if self.insightMode else 'MagPy4')
        self.app.setWindowIcon(self.marsIcon if self.insightMode else self.magpyIcon)

    def runTests(self):
        Tests.runTests()

    def openFileDialog(self, isFlatfile, clearCurrent):
        if isFlatfile:
            fileNames = QtWidgets.QFileDialog.getOpenFileNames(self, caption="Open Flat File", options = QtWidgets.QFileDialog.ReadOnly, filter='Flat Files (*.ffd)')[0]
        else:
            fileNames = QtWidgets.QFileDialog.getOpenFileNames(self, caption="Open CDF File", options = QtWidgets.QFileDialog.ReadOnly, filter='CDF Files (*.cdf)')[0]

        fileNames = list(fileNames)
        for fileName in fileNames:
            if '.' not in fileName: # lazy extension check
                print(f'Bad file found, cancelling open operation')
                return
        if not fileNames:
            print(f'No files selected, cancelling open operation')
            return

        if self.startUp:
            self.ui.setupView()
            self.startUp = False
            self.ui.showMMSMenu(not self.insightMode)
            self.setWindowTitle('MarsPy' if self.insightMode else 'MagPy4')

        if clearCurrent:
            for fid in self.FIDs:
                fid.close()
            self.FIDs = []
            self.initDataStorageStructures()

        for fileName in fileNames:
            if isFlatfile:        
                fileName = fileName.rsplit(".", 1)[0] #remove extension
                self.openFF(fileName)
            else:
                # temp solution because this still causes other problems
                # mainly need to add conversions for the different time formats
                try:
                    self.openCDF(fileName)
                except Exception as e:
                    print(e)
                    print('CDF LOAD FAILURE')

        self.closeAllSubWindows()
        self.initVariables()

        self.plotDataDefault()

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
        self.TIMES = [] # list of time informations (3 part lists) [time series, resolutions, average res]
        self.TIMEINDEX = {} # dict mapping dstrs to index into times list

        self.minTime = None # minimum time tick out of all loaded times
        self.maxTime = None # maximum
        self.iiE = None # maximum tick of sliders (min is always 0)
        self.resolution = 1000000.0 # this is set to minumum resolution when files are loaded so just start off as something large

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
        self.errorFlag = 1e7 # overriding for now since the above line is sometimes wrong depending on the file (i think bx saves as 1e31 but doesnt update header)
        print(f'error flag: {self.errorFlag:.0e}') # not being used currently
        #self.errorFlag *= 0.9 # based off FFSpectra.py line 829
        
        # load flatfile
        nRows = FID.getRows()
        records = FID.DID.sliceArray(row=1, nRow=nRows)
        ffTime = records["time"]
        self.dataByRec = records["data"]
        self.dataByCol = FF_File.arrayToColumns(records["data"])

        numRecords = len(self.dataByRec)
        numColumns = len(self.dataByCol)
        print(f'number records: {numRecords}')
        print(f'number columns: {numColumns}')

        datas = [np.array(col) for col in self.dataByCol]

        # ignoring first column because that is time, hence [1:]
        newDataStrings = FID.getColumnDescriptor("NAME")[1:]
        units = FID.getColumnDescriptor("UNITS")[1:]

        self.resolution = min(self.resolution,FID.getResolution())  # flatfile define resolution isnt always correct but whatever
        FID.getResolution() # u have to still call this otherwise ffsearch wont work and stuff

        # need to ensure loaded times are on same epoch, or do a conversion when plotting
        # loading files with same datastring names should either concatenate or just append a subnumber onto end

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
                    return

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

        # add file id to list

        self.FIDs.append(FID)

        self.calculateAbbreviatedDstrs()
        self.calculateTimeVariables()

        self.enableToolsAndOptionsMenus(True)

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

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.startSlider.underMouse():
            self.setTimes()

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

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.endSlider.underMouse():
            self.setTimes()

    def setSliderNoCallback(self, slider, i):
        slider.blockSignals(True)
        slider.setValue(i)
        slider.blockSignals(False)

    def onStartEditChanged(self, val):
        """this gets called when the start date time edit is changed directly"""
        tick = FFTIME(UTCQDate.QDateTime2UTC(val), Epoch=self.epoch)._tick
        self.iO = self.calcTickIndexByTime(tick)
        self.setSliderNoCallback(self.ui.startSlider, self.iO)
        for line in self.trackerLines:
            line.hide()
        self.setTimes()

    def onEndEditChanged(self, val):
        """this gets called when the end date time edit is changed directly"""
        tick = FFTIME(UTCQDate.QDateTime2UTC(val), Epoch=self.epoch)._tick
        self.iE = self.calcTickIndexByTime(tick)
        self.setSliderNoCallback(self.ui.endSlider, self.iE)
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

    # in seconds, abs for just incase backwards
    def getSelectedTimeRange(self):
        return abs(self.tE - self.tO)

    def getTimeLabelMode(self):
        rng = self.getSelectedTimeRange()
        if rng > self.dayCutoff: # if over day show MMM dd hh:mm:ss (don't need to label month and day)
            return 'DAY'
        elif rng > self.hrCutoff: # if over hour show hh:mm:ss
            return 'HR'
        elif rng > self.minCutoff: # if over 10 seconds show mm:ss
            return 'MIN'
        else: # else show mm:ss.sss
            return 'MS'

    def updateXRange(self):
        rng = self.getSelectedTimeRange()
        self.pltGrd.setTimeLabel('yellow')

        if rng > self.dayCutoff: # if over day show MMM dd hh:mm:ss (don't need to label month and day)
            self.pltGrd.setTimeLabel('HH:MM')
        elif rng > self.hrCutoff: # if hour show hh:mm:ss
            self.pltGrd.setTimeLabel('HH:MM:SS')
        elif rng > self.minCutoff: # if over 10 seconds show mm:ss
            self.pltGrd.setTimeLabel('MM:SS')
        else: # else show mm:ss.sss
            self.pltGrd.setTimeLabel('MM:SS.SSS')

        for pi in self.plotItems:
            pi.setXRange(self.tO-self.tickOffset, self.tE-self.tickOffset, 0.0)

        # Update ticks/labels on bottom axis, clear top axis
        labelMode = self.getTimeLabelMode()
        lastPlotIndex = len(self.plotItems) - 1
        lastPlot = self.plotItems[lastPlotIndex]
        tickAxis = lastPlot.getAxis('bottom')
        tickAxis.updateTicks(self, labelMode)
        topTicks = [(tv, '') for tv, ts in tickAxis._tickLevels[0]]
        lastPlot.getAxis('top').setTicks([topTicks,[]])

        # Update the other plot's tick marks to match (w/o labels)
        for i in range(0, lastPlotIndex):
            self.plotItems[i].getAxis('bottom').setTicks(tickAxis._tickLevels)
            self.plotItems[i].getAxis('top').setTicks([topTicks,[]])

        # Update additional tick labels, if there are any
        if self.pltGrd.labelSetGrd:
            self.pltGrd.labelSetGrd.updateTicks(tickAxis._tickLevels)

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
            if row:
                dstrs.append(row)
                links.append(ki)

        if not links: # if empty then at least show some empty plots so its less confusing
            dstrs = [[],[],[]]
            links = [0,1,2] 

        return dstrs, [links]


    def plotDataDefault(self):
        dstrs,links = self.getDefaultPlotInfo()

        self.plotData(dstrs, links)

    def getData(self, dstr, editNumber=None):
        edits = self.DATADICT[dstr]
        i = self.currentEdit if editNumber is None else editNumber
        while len(edits[i]) == 0: # if empty list go back one
            i -= 1
        return edits[i]

    def getLabel(self, dstr, editNumber=None):
        edits = self.DATADICT[dstr]
        i = self.currentEdit if editNumber is None else editNumber
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
        elif self.cdfName:
            name = self.cdfName
        return name

    def getMaxLabelWidth(self, label, gwin):
        # Gives the maximum number of characters that should, on average, fit
        # within the width of the window the label is in
        lblFont = label.font()
        fntMet = QtGui.QFontMetrics(lblFont)
        avgCharWidth = fntMet.averageCharWidth()
        winWidth = gwin.width()
        return winWidth / avgCharWidth

    def plotData(self, dataStrings, links):
        self.ui.glw.clear()
        self.closeTraceStats() # Clear any regions

        # save what the last plotted strings and links are for other modules
        self.lastPlotStrings = dataStrings
        self.lastPlotLinks = links

        self.plotItems = []
        self.labelItems = []

        # a list of pens for each trace (saved for consistency with spectra)
        self.plotTracePens = []

        # add label for file name at top right
        fileNameLabel = BLabelItem()
        fileNameLabel.opts['justify'] = 'right'
        maxLabelWidth = self.getMaxLabelWidth(fileNameLabel, self.ui.glw)
        fileNameLabel.setHtml(f"<span style='font-size:10pt;'>{self.getFileNameString(maxLabelWidth)}</span>")
        self.ui.glw.addItem(fileNameLabel, 0, 0, 1, 1)

        # Store any previous label sets (for current file)
        prevLabelSets = []
        if self.pltGrd is not None and self.pltGrd.labelSetGrd is not None:
            prevLabelSets = self.pltGrd.labelSetLabel.dstrs

        # Create new plot grid
        self.pltGrd = PlotGrid(self)
        self.ui.glw.addItem(self.pltGrd, 1, 0, 1, 1)

        self.trackerLines = []

        for plotIndex, dstrs in enumerate(dataStrings):
            axis = DateAxis(orientation='bottom')
            axis.window = self
            vb = MagPyViewBox(self, plotIndex)
            pi = pg.PlotItem(viewBox = vb, axisItems={'bottom': axis, 'left': LinkedAxis(orientation='left') })
            #pi.setClipToView(True) # sometimes cuts off part of plot so kinda trash?
            self.plotItems.append(pi) #save it for ref elsewhere
            vb.enableAutoRange(x=False, y=False) # range is being set manually in both directions
            pi.setDownsampling(ds=1, auto=True, mode='peak')

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
            #ba.style['textFillLimits'] = [(0,1.1)]
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
            dstrLabels = []
            dstrList = []
            colorsList = []
            # add traces on this plot for each dstr
            for i,(dstr,editNum) in enumerate(dstrs):
                # figure out which pen to use
                numPens = len(self.pens)
                if len(dstrs) == 1: # if just one trace then base it off which plot
                    penIndex = plotIndex % numPens
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

                dstrText = self.getLabel(dstr, editNum)
                if dstrText in self.ABBRV_DSTR_DICT:
                    dstrText = self.ABBRV_DSTR_DICT[dstrText]
                dstrLabels.append(dstrText)

                self.plotTrace(pi, dstr, editNum, pen)
                dstrList.append(dstr)
                colorsList.append(pen.color().name())

            self.plotTracePens.append(tracePens)

            # set plot to current range based on time sliders
            pi.setXRange(self.tO-self.tickOffset, self.tE-self.tickOffset, 0.0)

            #todo: needs to be set for current min and max of all time ranges
            pi.setLimits(xMin=self.minTime-self.tickOffset, xMax=self.maxTime-self.tickOffset)

            # Determine units to be placed on label
            unit = ''
            for dstr in dstrList:
                u = self.UNITDICT[dstr]
                # figure out if each axis trace shares same unit
                if unit == '':
                    unit = u
                elif unit != None and unit != u:
                    unit = None
            if unit == '':
                unit = None

            # Create plot label and add to grid
            stckLbl = StackedLabel(dstrLabels, colorsList, unit)
            self.pltGrd.addPlt(pi, stckLbl)

        ## end of main for loop

        # Add in all previous label sets, if there are any
        for labelSetDstr in prevLabelSets:
            self.pltGrd.addLabelSet(labelSetDstr)

        self.updateXRange()
        self.updateYRange()

        for pi in self.plotItems:
            for item in pi.items:
                item.viewRangeChanged()

    def replotDataCallback(self):
        # done this way to ignore the additional information ui callbacks will provide
        self.replotData()
        
    def replotData(self, desiredEdit=None):
        """simply redraws the traces and ensures y range is correct without rebuilding everything
           if desiredEdit is defined it will try to plot strings at that edit if they have data there
        """

        for i in range(len(self.plotItems)):
            pi = self.plotItems[i]
            plotStrs = self.lastPlotStrings[i]
            pens = self.plotTracePens[i]
            pi.clearPlots()

            # keep track of the frequency of strings in each plot (regardless of edit number)
            if desiredEdit is not None:
                seenCount = {} # how many times this string has been seen
                for dstr,en in plotStrs:
                    if dstr in seenCount:
                        seenCount[dstr]+=1
                    else:
                        seenCount[dstr] = 1

            j = 0
            while j < len(plotStrs):
                dstr,editNum = plotStrs[j]

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

                j+=1

        self.updateYRange()

    def getTimes(self, dstr, editNumber):
        times,resolutions,avgRes = self.TIMES[self.TIMEINDEX[dstr]]

        # check if arrays arent same length then assume the difference is from a filter operation
        Y = self.getData(dstr, editNumber)
        if len(Y) < len(times):
            diff = len(times) - len(Y) + 1
            times = times[diff // 2:-diff // 2 + 1]
            assert len(Y) == len(times), 'filter time correction failed...'
            resolutions = np.diff(times)
        return times,resolutions,avgRes

    # both plotData and replot use this function internally
    def plotTrace(self, pi, dstr, editNumber, pen):
        Y = self.getData(dstr, editNumber)
        if len(Y) <= 1: # not sure if this can happen but just incase
            print(f'Error: insufficient Y data for column "{dstr}"')
            return

        times,resolutions,avgRes = self.getTimes(dstr, editNumber)

        # Find smallest tick in data
        ofst = self.minTime
        self.tickOffset = ofst
        pi.tickOffset = ofst
        pi.getAxis('bottom').tickOffset = ofst
        # Subtract offset value from all times values for plotting
        ofstTimes = times - ofst

        # Determine data segments/type and plot
        if not self.ui.bridgeDataGaps.isChecked():
            # Replace error flags with NaN so those points will not be plotted
            YWithNan = Mth.replaceErrorsWithNaN(Y, self.ORIGDATADICT[dstr], self.errorFlag)
            # Split data into segments so points with large time gaps are not connected
            segs = Mth.getSegmentsFromTimeGaps(resolutions, avgRes*2)
            for a, b in segs:
                if self.ui.drawPoints.isChecked():
                    pi.addItem(PlotPointsItem(ofstTimes[a:b], YWithNan[a:b], pen=pen, connect='finite'))
                else:
                    pi.addItem(PlotDataItemBDS(ofstTimes[a:b], YWithNan[a:b], pen=pen, connect='finite'))
        else:
            if self.ui.drawPoints.isChecked():
                pi.addItem(PlotPointsItem(ofstTimes, Y, pen=pen))
            else:
                pi.addItem(PlotDataItemBDS(ofstTimes, Y, pen=pen))
    
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
            # find min and max values out of all traces on this plot
            for (dstr,editNum) in dstrs:
                Y = self.getData(dstr,editNum)
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

                minVal = min(minVal, Y.min())
                maxVal = max(maxVal, Y.max())
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
                if i in skipRangeSet:
                    continue
                diff = values[i][1] - values[i][0]
                largest = max(largest,diff)

            # then scale each plot in this row to the range
            for i in row:
                if i in skipRangeSet:
                    continue
                diff = values[i][1] - values[i][0]
                l2 = (largest - diff) / 2.0
                self.plotItems[i].setYRange(values[i][0] - l2, values[i][1] + l2, padding = 0.05)

    def updateTraceStats(self):
        if self.traceStats:
            self.traceStats.update()

    def updateCurlometer(self):
        if self.curlometer:
            self.curlometer.calculate()

    # color is hex string ie: '#ff0000' for red
    def initGeneralSelect(self, name, color, timeEdit, canHide=False):
        # Initialize all variables used to set up selected regions
        self.selectMode = name
        self.selectTimeEdit = timeEdit
        self.selectColor = color
        if timeEdit is not None:
            timeEdit.linesConnected = False

    def connectLinesToTimeEdit(self, timeEdit, region, single=False):
        if timeEdit == None:
            return
        elif single:
            timeEdit.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, region))
            return
        if self.selectMode == 'Stats' and timeEdit.linesConnected:
            # Disconnect from any previously connected regions (only in Stats mode)
            timeEdit.start.dateTimeChanged.disconnect()
            timeEdit.end.dateTimeChanged.disconnect()
        # Connect timeEdit to currently being moved / recently added region
        timeEdit.start.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, region))
        timeEdit.end.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, region))
        timeEdit.linesConnected = True

    def endGeneralSelect(self):
        # Clear all region selection variables
        self.selectTimeEdit = None
        self.selectMode = None
        self.selectColor = None

        # Remove all selected regions from actual plots
        for region in self.regions:
            region.removeRegionItems()
        self.regions = []

    def updateLinesByTimeEdit(self, timeEdit, region, single=False):
        x0, x1 = region.getRegion()
        i0, i1 = self.getTicksFromTimeEdit(timeEdit)
        t0 = self.getTimeFromTick(i0)
        t1 = self.getTimeFromTick(i1)
        if self.selectMode == 'Curlometer':
            i0 = self.calcTickIndexByTime(FFTIME(UTCQDate.QDateTime2UTC(timeEdit.start.dateTime()), Epoch=self.epoch)._tick)
            t0 = self.getTimeFromTick(i0)
            self.updateLinesPos(region, t0, t0)
            return
        assert(t0 <= t1)
        self.updateLinesPos(region, t0 if x0 < x1 else t1, t1 if x0 < x1 else t0)

    def updateTimeEditByLines(self, timeEdit, region):
        x0, x1 = region.getRegion()
        t0 = UTCQDate.UTC2QDateTime(FFTIME(x0, Epoch=self.epoch).UTC)
        t1 = UTCQDate.UTC2QDateTime(FFTIME(x1, Epoch=self.epoch).UTC)

        timeEdit.setStartNoCallback(min(t0,t1))
        timeEdit.setEndNoCallback(max(t0,t1))

    def updateLinesPos(self, region, t0, t1):
        region.setRegion((t0 - self.tickOffset, t1 - self.tickOffset))
        self.updateTraceStats()
        self.updateCurlometer()

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

    # given the corresponding time array for data (times) and the time (t), calculate index into data array
    def calcDataIndexByTime(self, times, t):
        assert(len(times) >= 2)
        if t <= times[0]:
            return 0
        if t >= times[-1]:
            return len(times)
        b = bisect.bisect_left(times, t) # can bin search because times are sorted
        assert(b)
        return b

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
        assert(i0 <= i1)
        return i0,i1

    def getSelectionStartEndTimes(self, regNum=0):
        if self.regions == []:
            return self.minTime, self.maxTime
        t0, t1 = self.regions[regNum].getRegion()
        return (t0,t1) if t0 <= t1 else (t1,t0) # need parens here!

    def getSelectedPlotInfo(self):
        """based on which plots have active lines, return list for each plot of the datastr and pen for each trace"""

        plotInfo = []
        for i,pi in enumerate(self.plotItems):
            if pi.getViewBox().anyLinesVisible():
                plotInfo.append((self.lastPlotStrings[i], self.plotTracePens[i]))
        return plotInfo

# look at the source here to see what functions you might want to override or call
#http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ViewBox/ViewBox.html#ViewBox
class MagPyViewBox(pg.ViewBox): # custom viewbox event handling
    def __init__(self, window, plotIndex, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.window = window
        self.plotIndex = plotIndex
        self.menuSetup()

    def menuSetup(self):
        # Remove menu options that won't be used
        actions = self.menu.actions()
        xAction, yAction, mouseAction = actions[1:4]
        for a in [xAction, yAction, mouseAction]:
            self.menu.removeAction(a)
        # Add in custom menu actions
        self.menu.addAction(self.window.ui.plotApprAction) # Plot appearance
        self.menu.addAction(self.window.ui.addTickLblsAction) # Additional labels

    def onLeftClick(self, ev):
        # map the mouse click to data coordinates
        mc = self.mapToView(ev.pos())
        x = mc.x()
        y = mc.y()

        # if just clicking on the plots with no general select started then open a trace stats window
        if self.window.regions == [] and self.window.selectMode == None:
            self.window.openTraceStats(self.plotIndex)
            self.window.initGeneralSelect('Stats', None, self.window.traceStats.ui.timeEdit)

        window = self.window
        plts, mode, color = window.plotItems, window.selectMode, window.selectColor

        # Holding ctrl key in stats mode allows selecting multiple regions
        ctrlPressed = (ev.modifiers() == QtCore.Qt.ControlModifier)
        multiSelect = (ctrlPressed and mode == 'Stats')
        singleLineMode = (mode == 'Curlometer')

        # Case where no regions have been created or just adding a new line
        if window.regions == [] or (multiSelect and not window.regions[-1].isLine()):
            region = LinkedRegion(window, plts, values=(x, x), mode=mode, color=color)
            window.regions.append(region)
            if mode == 'Curlometer':
                self.window.connectLinesToTimeEdit(self.window.selectTimeEdit, region)
                QtCore.QTimer.singleShot(100, self.window.showCurlometer)
            else:
                # Initial connection to time edit
                self.window.connectLinesToTimeEdit(self.window.selectTimeEdit, region)

        # Case where last region added is still a line
        elif window.regions[-1].isLine() and not singleLineMode:
            # Get line's position x0 and create region between x0 and current x
            prevX = self.window.regions[-1].linePos()
            window.regions[-1].setRegion((prevX, x))

            if self.window.edit:
                PyQtUtils.moveToFront(self.window.edit.minVar)

            if mode == 'Spectra' and len(self.window.regions) == 1:
                QtCore.QTimer.singleShot(100, self.window.showSpectra)

            if mode == 'Plane Normal' and len(self.window.regions) == 1:
                QtCore.QTimer.singleShot(100, self.window.showNormal)

        # Case where sub-region was previously set to hidden
        elif window.regions[-1].isVisible(self.plotIndex) == False and not singleLineMode:
            # Make sub-regions visible for this plot again
            for region in window.regions:
                region.setVisible(True, self.plotIndex)

        # If this will be the only line in plots, drag shouldn't expand region
        numRegions = len(window.regions)
        if numRegions == 1 and window.regions[0].isLine():
            window.regions[0].fixedLine = True
        # Once first region is set, allow dragging the edges
        elif numRegions > 0 and not singleLineMode: 
            window.regions[0].fixedLine = False

        self.window.updateTraceStats()
        if mode != 'Plane Normal':
            self.window.updateTimeEditByLines(self.window.selectTimeEdit, self.window.regions[-1])

    # check if either of lines are visible for this viewbox
    def anyLinesVisible(self):
        isVisible = False
        for region in self.window.regions:
            if region.isVisible(self.plotIndex):
                isVisible = True
        return isVisible

    # sets the lines of this viewbox visible
    def setMyLinesVisible(self, isVisible):
        for region in self.window.regions:
            region.setVisible(isVisible, self.plotIndex)

    def onRightClick(self, ev):
        if self.anyLinesVisible(): # cancel selection on this plot (if able to)
            self.setMyLinesVisible(False)
        else:
            pg.ViewBox.mouseClickEvent(self,ev) # default right click

        if not self.window.getSelectedPlotInfo(): # no plots then close
            self.window.closeTraceStats()
            self.window.closeCurlometer()
            self.window.closePlaneNormal()
            for region in self.window.regions:
                region.removeRegionItems()
            self.window.regions = []
        else:
            self.window.updateTraceStats()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.double(): # double clicking will do same as right click
                #self.onRightClick(ev)
                pass # this seems to confuse people so disabling for now
            else:
               self.onLeftClick(ev)

        else: # asume right click i guess, not sure about middle mouse button click
            self.onRightClick(ev)

        ev.accept()
            
    # mouse drags for now just get turned into clicks, like on release basically, feels nicer
    # technically only need to do this for spectra mode but not being used otherwise so whatever
    def mouseDragEvent(self, ev, axis=None):
        if ev.isFinish(): # on release
            if ev.button() == QtCore.Qt.LeftButton:
                self.onLeftClick(ev)
            elif ev.button() == QtCore.Qt.RightButton:
                self.onRightClick(ev)
        ev.accept()
        #    pg.ViewBox.mouseDragEvent(self, ev)

    def wheelEvent(self, ev, axis=None):
        ev.ignore()


def myexepthook(type, value, tb):
    print(f'{type} {value}')
    traceback.print_tb(tb,limit=5)
    os.system('pause')

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    #appName = f'{appName} {version}';
    app.setOrganizationName('IGPP UCLA')
    app.setOrganizationDomain('igpp.ucla.edu')
    app.setApplicationName('MagPy4')
    #app.setApplicationVersion(version)

    main = MagPy4Window(app)
    main.showMaximized()

    sys.excepthook = myexepthook
    sys.exit(app.exec_())