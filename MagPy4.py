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
VERSION = f'Version 1.0.0.1 (December 4, 2018)'
COPYRIGHT = f'Copyright © 2018 The Regents of the University of California'

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
import pyqtgraph as pg

import FF_File
from FF_Time import FFTIME, leapFile

from MagPy4UI import MagPy4UI, PyQtUtils
from plotMenu import PlotMenu
from spectra import Spectra
from dataDisplay import DataDisplay, UTCQDate
from edit import Edit
from traceStats import TraceStats
from helpWindow import HelpWindow
from AboutDialog import AboutDialog
from pyqtgraphExtensions import DateAxis, LinkedAxis, PlotPointsItem, PlotDataItemBDS, LinkedInfiniteLine, BLabelItem
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
        self.insightMode = False

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
        self.edit = None
        self.traceStats = None
        self.helpWindow = None
        self.aboutDialog = None
        self.FIDs = []
        self.tickOffset = 0 # Smallest tick in data, used when plotting x data

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
        colors = ['#0000ff','#009900','#ff0000','#000000'] # b darkgreen r black
        for c in colors:
            self.pens.append(pg.mkPen(c, width=1))# style=QtCore.Qt.DotLine)
        self.trackerPen = pg.mkPen('#000000', width=1, style=QtCore.Qt.DashLine)

        self.plotItems = []
        self.labelItems = []
        self.trackerLines = []
        #starterFile = 'testData/mms15092720'
        starterFile = 'testData/insight/IFGlr_pCAL_20180816T045752_20180817T090012' #insight test file
        if os.path.exists(starterFile + '.ffd'):
            self.openFF(starterFile)
            self.swapMode()

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
        self.closeEdit()
        self.closeData()
        self.closeTraceStats()
        self.closeSpectra()

    def initVariables(self):
        """init variables here that should be reset when file changes"""
        self.lastPlotStrings = None
        self.lastPlotLinks = None
        # 0: not selecting, 1 : no lines, 2 : one line, 3+ : two lines
        self.generalSelectStep = 0
        self.generalSelectCanHide = False
        self.currentEdit = 0 # current edit number selected
        self.editNames = [] # list of edit names, index into list is edit number
        self.editHistory = []
        
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

    def openPlotMenu(self):
        self.closePlotMenu()
        self.plotMenu = PlotMenu(self)

        geo = self.geometry()
        self.plotMenu.move(geo.x() + 200, geo.y() + 100)
        self.plotMenu.show()

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
            self.startGeneralSelect('SPECTRA', '#FF0000', self.spectra.ui.timeEdit, True)

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
        tooltip = 'Loads various presets specific to the MMS mission and better for general use cases' if self.insightMode else 'Loads various presets specific to the Insight mission'
        self.ui.switchMode.setText(txt)
        self.ui.switchMode.setToolTip(tooltip)
        self.plotDataDefault()
        self.setWindowTitle('MarsPy' if self.insightMode else 'MagPy4')
        self.app.setWindowIcon(self.marsIcon if self.insightMode else self.magpyIcon)

    def runTests(self):
        Tests.runTests()

    def resizeEvent(self, event):
        #print(event.size())
        #self.additionalResizing()
        pass

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

        #self.resolution = min(self.resolution,FID.getResolution())  # flatfile define resolution isnt always correct but whatever
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
            segments = Mth.getSegmentsFromTimeGaps(curRes, curAvgRes * 2)
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
        self.ui.timeLabel.setText('yellow')

        if rng > self.dayCutoff: # if over day show MMM dd hh:mm:ss (don't need to label month and day)
            self.ui.timeLabel.setText('hh:mm:ss')
        elif rng > self.hrCutoff: # if hour show hh:mm:ss
            self.ui.timeLabel.setText('hh:mm:ss')
        elif rng > self.minCutoff: # if over 10 seconds show mm:ss
            self.ui.timeLabel.setText('mm:ss')
        else: # else show mm:ss.sss
            self.ui.timeLabel.setText('mm:ss.sss')

        for pi in self.plotItems:
            pi.setXRange(self.tO-self.tickOffset, self.tE-self.tickOffset, 0.0)

        # Update ticks/labels on bottom axis, clear top axis
        labelMode = self.getTimeLabelMode()
        lastPlotIndex = len(self.plotItems) - 1
        lastPlot = self.plotItems[lastPlotIndex]
        tickAxis = lastPlot.getAxis('bottom')
        tickAxis.updateTicks(self, labelMode)
        lastPlot.getAxis('top').setTicks([[],[]])

        # Update the other plot's tick marks to match (w/o labels)
        for i in range(0, lastPlotIndex):
            self.plotItems[i].getAxis('bottom').setTicks(tickAxis._tickLevels)
            self.plotItems[i].getAxis('top').setTicks([[],[]])

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

    def getFileNameString(self): # returns list of all loaded files
        name = 'unknown'
        if len(self.FIDs) > 1:
            names = [os.path.split(FID.name)[1] for FID in self.FIDs]
            if len(names) > 4: # abbreviate if too many files are loaded
                del[names[3:len(names) - 1]]
                names.insert(3,'... ')
            name = ', '.join(names)
        elif len(self.FIDs) > 0:
            name = self.FIDs[0].name
        elif self.cdfName:
            name = self.cdfName
        return name

    def plotData(self, dataStrings, links):
        self.ui.glw.clear()

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
        fileNameLabel.setHtml(f"<span style='font-size:10pt;'>{self.getFileNameString()}</span>")
        self.ui.glw.nextColumn()
        self.ui.glw.addItem(fileNameLabel)
        self.ui.glw.nextRow()

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
            # add traces on this plot for each dstr
            for i,(dstr,editNum) in enumerate(dstrs):
                u = self.UNITDICT[dstr]

                # figure out which pen to use
                numPens = len(self.pens)
                if len(dstrs) == 1: # if just one trace then base it off which plot
                    penIndex = plotIndex % numPens
                else: # else if base off trace index, capped at pen count
                    penIndex = min(i,numPens - 1) 
                pen = self.pens[penIndex]

                #save pens so spectra can stay synced with main plot
                tracePens.append(pen)

                self.plotTrace(pi, dstr, editNum, pen)

            self.plotTracePens.append(tracePens)

            # set plot to current range based on time sliders
            pi.setXRange(self.tO-self.tickOffset, self.tE-self.tickOffset, 0.0)

            #todo: needs to be set for current min and max of all time ranges
            pi.setLimits(xMin=self.minTime-self.tickOffset, xMax=self.maxTime-self.tickOffset)

            # add Y axis label based on traces (label gets set below)
            li = BLabelItem()
            self.labelItems.append(li)

            self.ui.glw.addItem(li)
            self.ui.glw.addItem(pi)
            self.ui.glw.nextRow()

        self.additionalResizing()

        ## end of main for loop

        self.updateXRange()
        self.updateYRange()

        for pi in self.plotItems:
            for item in pi.items:
                item.viewRangeChanged()

    ## end of plot function

    def setYAxisLabels(self):
        """sets y axis label strings for each plot"""
        plots = len(self.plotItems)
        for dstrs,pens,li in zip(self.lastPlotStrings,self.plotTracePens,self.labelItems):
            traceCount = len(dstrs)
            alab = ''
            unit = ''
            for (dstr,editNum),pen in zip(dstrs,pens):
                u = self.UNITDICT[dstr]
                # figure out if each axis trace shares same unit
                if unit == '':
                    unit = u
                elif unit != None and unit != u:
                    unit = None

                l = self.getLabel(dstr, editNum)
                if l in self.ABBRV_DSTR_DICT:
                    l = self.ABBRV_DSTR_DICT[l]
                alab += f"<span style='color:{pen.color().name()};'>{l}</span>\n"

            # add unit label if each trace on this plot shares same unit
            if unit != None and unit != '':
                alab += f"<span style='color:#888888;'>[{unit}]</span>\n"
            else:
                alab = alab[:-1] #remove last newline character

            fontSize = self.suggestedFontSize
            if traceCount > plots and plots > 1:
                fontSize -= (traceCount - plots) * (1.0 / min(4, plots) + 0.35)
            fontSize = min(16, max(fontSize,4))
            li.setHtml(f"<span style='font-size:{fontSize}pt; white-space:pre;'>{alab}</span>")


    def additionalResizing(self):
        """
        this function tries to correctly estimate size of rows. the problem is last one needs to be a bit larger since bottom axis
        this is super hacked but it actually works okay. end goal is for bottom plot to be same height as the rest
        """

        # may want to redo with viewGeometry of plots in mind, might be more consistent than fontsize stuff on mac for example
        #for pi in self.plotItems:
        #    print(pi.viewGeometry())

        #print('additionally resizing')

        qggl = self.ui.glw.layout
        rows = qggl.rowCount()
        plots = rows - 1

        width = self.ui.glw.viewRect().width()
        height = self.ui.glw.viewRect().height()

        self.ui.timeLabel.setPos(width / 2, height - 30)

        # set font size, based on viewsize and plot number
        # very hardcoded just based on nice numbers i found. the goal here is for the labels to be smaller so the plotItems will dictate scaling
        # if these labelitems are larger they will cause qgridlayout to stretch and the plots wont be same height which is bad
        if plots > 0:
            self.suggestedFontSize = height / plots * 0.065
            self.setYAxisLabels()

        if plots <= 1: # if just one plot dont need handle multiplot problems
            return

        # all of this depends on screen resolution so this is a pretty bad fix for now
        bPadding = 20 # for bottom axis text
        #plotSpacing = 10 if self.OS == 'mac' else 7
        plotSpacing = 0.01 * height   #7/689 is about 1% of screen, this is so filth lol but it kinda works

        edgePadding = 60
        height -= bPadding + edgePadding + (plots - 1) * plotSpacing # the spaces in between plots hence the -1

        # when running out of room bottom plot starts shrinking
        # pyqtgraph is doing some additional resizing independent of the underlying qt layout, but cant figure out where (this kind of works)
        for i in range(rows):
            if i == 0:
                continue

            h = height / plots if i < rows - 1 else height / plots + bPadding
            qggl.setRowFixedHeight(i, h)
            qggl.setRowMinimumHeight(i, h)
            qggl.setRowMaximumHeight(i, h) 

        # this ensures the plots always have same physical width (important because they all share bottom time axis)
        # it works but at start of plot its wrong because the bounds and stuff isnt correct yet
        # this is a recurring problem in multiple areas but not sure how to figure out yet and its not that bad in this case
        maxWidth = 0
        for pi in self.plotItems:
            la = pi.getAxis('left')
            maxWidth = max(maxWidth, la.calcDesiredWidth())
        for pi in self.plotItems:
            la = pi.getAxis('left')
            la.setWidth(maxWidth)
        #print(maxWidth)

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

        self.setYAxisLabels()
        self.updateYRange()

    def getTimes(self, dstr, editNumber):
        times,resolutions,avgRes = self.TIMES[self.TIMEINDEX[dstr]]

        # check if arrays arent same length then assume the difference is from a filter operation
        Y = self.getData(dstr, editNumber)
        if len(Y) < len(times):
            diff = len(times) - len(Y) + 1
            times = times[diff // 2:-diff // 2 + 1]
            # resolutions shouldn't change because they are used with original data only
            assert len(Y) == len(times), 'filter time correction failed...'

        return times,resolutions,avgRes

    # both plotData and replot use this function internally
    def plotTrace(self, pi, dstr, editNumber, pen):
        Y = self.getData(dstr, editNumber)
        if len(Y) <= 1: # not sure if this can happen but just incase
            print(f'Error: insufficient Y data for column "{dstr}"')
            return

        times,resolutions,avgRes = self.getTimes(dstr, editNumber)

        # Find smallest tick in data
        ofst = min(times)
        self.tickOffset = ofst
        pi.tickOffset = ofst
        pi.getAxis('bottom').tickOffset = ofst
        # Subtract offset value from all times values for plotting
        ofstTimes = [t - ofst for t in times]

        # Determine data segments/type and plot
        if not self.ui.bridgeDataGaps.isChecked():
            segs = Mth.getSegmentsFromErrorsAndGaps(self.ORIGDATADICT[dstr], resolutions, self.errorFlag, avgRes * 2)   
            for a,b in segs:
                if self.ui.drawPoints.isChecked():
                    pi.addItem(PlotPointsItem(ofstTimes[a:b], Y[a:b], pen=pen))
                else:
                    pi.addItem(PlotDataItemBDS(ofstTimes[a:b], Y[a:b], pen=pen))
        else:
            if self.ui.drawPoints.isChecked():
                pi.addItem(PlotPointsItem(ofstTimes, Y, pen=pen))
            else:
                pi.addItem(PlotDataItemBDS(ofstTimes, Y, pen=pen))

        # Get the plot's general lines and set its bounds/tickOffset
        lines = pi.getViewBox().lines
        for k in lines:
            for i in lines[k]:
                i.tickOffset = self.tickOffset
                i.setBounds((self.minTime-self.tickOffset, self.maxTime-self.tickOffset))
    
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

    # color is hex string ie: '#ff0000' for red
    def startGeneralSelect(self, name, color, timeEdit, canHide=False):
        self.updateLinesAppearance(name, color)
        self.generalSelectStep = 1
        self.generalSelectCanHide = canHide
        self.generalTimeEdit = timeEdit
        self.resetLinesPos('general')
        self.connectLinesToTimeEdit(timeEdit, 'general')

    def connectLinesToTimeEdit(self, timeEdit, lineStr):
        timeEdit.start.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, lineStr))
        timeEdit.end.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, lineStr))

    def endGeneralSelect(self):
        self.generalSelectStep = 0
        self.generalTimeEdit = None
        self.setLinesVisible(False, 'general')

    def updateLinesByTimeEdit(self, timeEdit, lineStr):
        x0 = self.plotItems[0].getViewBox().lines[lineStr][0].getXOfstPos()
        x1 = self.plotItems[0].getViewBox().lines[lineStr][1].getXOfstPos()
        i0,i1 = self.getTicksFromTimeEdit(timeEdit)
        t0 = self.getTimeFromTick(i0)
        t1 = self.getTimeFromTick(i1)
        assert(t0 <= t1)
        self.updateLinesPos(lineStr, 0, t0 if x0 < x1 else t1)
        self.updateLinesPos(lineStr, 1, t1 if x0 < x1 else t0)

    def updateTimeEditByLines(self, timeEdit, lineStr, index):
        oi = 0 if index == 1 else 1
        x = self.plotItems[0].getViewBox().lines[lineStr][index].getXOfstPos()
        ox = self.plotItems[0].getViewBox().lines[lineStr][oi].getXOfstPos()
        t0 = UTCQDate.UTC2QDateTime(FFTIME(x, Epoch=self.epoch).UTC)
        t1 = UTCQDate.UTC2QDateTime(FFTIME(ox, Epoch=self.epoch).UTC)

        timeEdit.setStartNoCallback(min(t0,t1))
        timeEdit.setEndNoCallback(max(t0,t1))

        self.updateLinesPos(lineStr, index, x)

    def updateGeneralLines(self, index, x):
        # General lines use the true line position (not the tick time from data),
        # so it must be set directly
        for pi in self.plotItems:
            pi.getViewBox().lines['general'][index].setPos(x)
        self.updateTimeEditByLines(self.generalTimeEdit, 'general', index)

    def updateLinesPos(self, lineStr, index, x):
        for pi in self.plotItems:
            pi.getViewBox().lines[lineStr][index].setOfstPos(x)
        self.updateTraceStats()

    def resetLinesPos(self, lineStr):
        self.updateLinesPos(lineStr, 0, self.minTime)
        self.updateLinesPos(lineStr, 1, self.maxTime)

    def setLinesVisible(self, isVisible, lineStr, index=None):
        #print(f'{isVisible} {lineStr} {index} {exc}')
        for pi in self.plotItems:
            vb = pi.getViewBox()
            if index is None:
                for line in vb.lines[lineStr]:
                    line.setVisible(isVisible)
            else:
                vb.lines[lineStr][index].setVisible(isVisible)      

    def matchLinesVisible(self, lineStr):
        for pi in self.plotItems:
            vb = pi.getViewBox()
            vb.lines[lineStr][1].setVisible(vb.lines[lineStr][0].isVisible())


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
    def calcDataIndicesFromLines(self, dstr, editNumber):
        """given a data string, calculate its indices based on time range currently selected with lines"""

        times = self.getTimes(dstr,editNumber)[0]
        t0,t1 = self.getSelectionStartEndTimes()
        i0 = self.calcDataIndexByTime(times, t0)
        i1 = self.calcDataIndexByTime(times, t1)
        if i1 > len(times)-1:
            i1 = len(times)-1
        assert(i0 <= i1)
        return i0,i1

    def getSelectionStartEndTimes(self):
        lines = self.plotItems[0].getViewBox().lines['general']
        t0 = lines[0].getXOfstPos()
        t1 = lines[1].getXOfstPos()
        return (t0,t1) if t0 <= t1 else (t1,t0) # need parens here!

    def getSelectedPlotInfo(self):
        """based on which plots have active lines, return list for each plot of the datastr and pen for each trace"""

        plotInfo = []
        for i,pi in enumerate(self.plotItems):
            if pi.getViewBox().lines['general'][0].isVisible():
                plotInfo.append((self.lastPlotStrings[i], self.plotTracePens[i]))
        return plotInfo

    # show label on topmost line pair
    def updateLineTextPos(self):
        foundFirst = False
        for i,pi in enumerate(self.plotItems):
            lines = pi.getViewBox().lines['general']
            if not foundFirst and lines[0].isVisible():
                lines[0].mylabel.show()
                if lines[1].isVisible():
                    lines[1].mylabel.show()
                foundFirst = True
            else:
                lines[0].mylabel.hide()
                lines[1].mylabel.hide()

    def updateLinesAppearance(self, name, color):
        pen = pg.mkPen(color, width=1, style=QtCore.Qt.DashLine)
        for i,pi, in enumerate(self.plotItems):
            lines = pi.getViewBox().lines['general']
            for line in lines:
                line.setPen(pen)
                line.mylabel.setText(name, color)
        

# look at the source here to see what functions you might want to override or call
#http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ViewBox/ViewBox.html#ViewBox
class MagPyViewBox(pg.ViewBox): # custom viewbox event handling
    def __init__(self, window, plotIndex, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #self.setMouseMode(self.RectMode)
        self.window = window
        self.plotIndex = plotIndex

        generalLines = []
        for i in range(2):
            generalLines.append(LinkedInfiniteLine(functools.partial(window.updateGeneralLines, i), movable=True, angle=90, pos=0, mylabel='GENERAL', labelColor='#000000'))
        self.lines = {'general':generalLines} # dictionary like this incase want to have concurrent line sets later
        for key,lines in self.lines.items():
            for line in lines:
                self.addItem(line, ignoreBounds = True)
                line.setBounds((self.window.minTime, self.window.maxTime))
                line.hide()

    def onLeftClick(self, ev):
         # map the mouse click to data coordinates
        mc = self.mapToView(ev.pos())
        x = mc.x()
        y = mc.y()
        #print(f'{x} {y}')

        # if just clicking on the plots with no general select started then open a trace stats window
        if self.window.generalSelectStep == 0:
            self.window.openTraceStats(self.plotIndex)
            self.window.startGeneralSelect('STATS', '#009900', self.window.traceStats.ui.timeEdit, True)

        # if deselected everything then go back to first step
        if self.window.generalSelectStep >= 3 and not self.window.getSelectedPlotInfo():
            self.window.generalSelectStep = 1

        # add the first or second lines
        if self.window.generalSelectStep > 0 and self.window.generalSelectStep < 3:
            if self.window.generalSelectStep == 1:
                self.window.generalSelectStep += 1
                self.window.setLinesVisible(True, 'general', 0)
                self.window.updateGeneralLines(0,x)
            elif self.window.generalSelectStep == 2:
                self.window.generalSelectStep += 1
                self.window.matchLinesVisible('general')
                self.window.updateGeneralLines(1,x)
                if self.window.edit:
                    PyQtUtils.moveToFront(self.window.edit.minVar)
    
                QtCore.QTimer.singleShot(100, self.window.showSpectra) #calls it with delay so ui has chance to draw lines first

            self.window.updateLineTextPos()
            
        # reselect this plot if it was deselected
        elif self.window.generalSelectStep >= 3 and not self.anyLinesVisible():
            self.setMyLinesVisible(True)

        self.window.updateTraceStats()

    # check if either of lines are visible for this viewbox
    def anyLinesVisible(self):
        lines = self.lines['general']
        return lines[0].isVisible() or lines[1].isVisible()

    # sets the lines of this viewbox visible
    def setMyLinesVisible(self, isVisible):
        for line in self.lines['general']:
            line.setVisible(isVisible)
        self.window.updateLineTextPos()

    def onRightClick(self, ev):
        if self.window.generalSelectStep > 1 and self.window.generalSelectCanHide: # cancel selection on this plot (if able to)
            self.setMyLinesVisible(False)
        else:
            pg.ViewBox.mouseClickEvent(self,ev) # default right click

        if not self.window.getSelectedPlotInfo(): # no plots then close
            self.window.closeTraceStats()
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