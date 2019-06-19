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

from MagPy4UI import MagPy4UI, PyQtUtils, MainPlotGrid, StackedLabel, TimeEdit
from plotMenu import PlotMenu
from spectra import Spectra
from dataDisplay import DataDisplay, UTCQDate
from plotAppearance import MagPyPlotApp
from addTickLabels import AddTickLabels
from edit import Edit
from traceStats import TraceStats
from helpWindow import HelpWindow
from AboutDialog import AboutDialog
from pyqtgraphExtensions import DateAxis, LinkedAxis, PlotPointsItem, PlotDataItemBDS, BLabelItem, LinkedRegion, MagPyPlotItem
from MMSTools import PlaneNormal, Curlometer, Curvature, ElectronPitchAngle, ElectronOmni
from detrendWin import DetrendWindow
from dynamicSpectra import DynamicSpectra, DynamicCohPha
from waveAnalysis import DynamicWave
from smoothingTool import SmoothingTool
from ffCreator import createFF
from mth import Mth
from tests import Tests
import bisect
from timeManager import TimeManager
from selectionManager import GeneralSelect
from layoutTools import BaseLayout

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
        self.ui.actionExportFF.triggered.connect(self.exportFlatFile)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionShowData.triggered.connect(self.showData)
        self.ui.actionPlotMenu.triggered.connect(self.openPlotMenu)
        self.ui.actionSpectra.triggered.connect(self.startSpectra)
        self.ui.actionDynamicSpectra.triggered.connect(self.startDynamicSpectra)
        self.ui.actionDynamicCohPha.triggered.connect(self.startDynamicCohPha)
        self.ui.actionDynWave.triggered.connect(self.startDynWave)
        self.ui.actionDetrend.triggered.connect(self.startDetrend)
        self.ui.actionEdit.triggered.connect(self.openEdit)
        self.ui.actionHelp.triggered.connect(self.openHelp)
        self.ui.actionAbout.triggered.connect(self.openAbout)
        self.ui.switchMode.triggered.connect(self.swapMode)
        self.ui.runTests.triggered.connect(self.runTests)

        # MMS Tool actions
        self.ui.actionPlaneNormal.triggered.connect(self.openPlaneNormal)
        self.ui.actionCurlometer.triggered.connect(self.openCurlometer)
        self.ui.actionCurvature.triggered.connect(self.openCurvature)
        self.ui.actionEPAD.triggered.connect(self.startEPAD)
        self.ui.actionEOmni.triggered.connect(self.startEOMNI)

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
        self.dynSpectra = None
        self.dynCohPha = None
        self.dynWave = None
        self.plotAppr = None
        self.addTickLbls = None
        self.edit = None
        self.traceStats = None
        self.helpWindow = None
        self.aboutDialog = None
        self.FIDs = []
        self.tickOffset = 0 # Smallest tick in data, used when plotting x data
        self.smoothing = None
        self.detrendWin = None
        self.currSelect = None

        # MMS Tools
        self.planeNormal = None
        self.curlometer = None
        self.curvature = None
        self.electronPAD = None
        self.electronOMNI = None

        # these are saves for options for program lifetime
        self.plotMenuTableMode = False
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
        self.closeSmoothing()
        self.closeAddTickLbls()
        self.closePlaneNormal()
        self.closeDynamicSpectra()
        self.closeDynamicCohPha()
        self.closeDynWave()
        self.closeMMSTools()
        self.closeDetrend()

    def closePlotTools(self):
        self.closeDetrend()
        self.closeSpectra()
        self.closeDynamicCohPha()
        self.closeDynamicSpectra()
        self.closeTraceStats()

    def initVariables(self):
        """init variables here that should be reset when file changes"""
        self.lastPlotStrings = None
        self.lastPlotLinks = None
        self.selectMode = None
        self.currentEdit = 0 # current edit number selected
        self.editNames = [] # list of edit names, index into list is edit number
        self.editHistory = []
        self.changeLog = {}
        self.customPens = []
        self.pltGrd = None
        self.regions = []
        self.newVars = []
        
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

    def closeMMSTools(self):
        self.closePlaneNormal()
        self.closeCurlometer()
        self.closeCurvature()
        self.closeEPAD()
        self.closeEOMNI()

    def startEOMNI(self):
        self.closeEOMNI()
        if self.electronOMNI is None or self.electronOMNI.wasClosed:
            self.electronOMNI = ElectronOmni(self)
            self.initGeneralSelect('Electron/Ion Spectrum', None, self.electronOMNI.ui.timeEdit,
            'Single', self.showEOMNI, closeFunc=self.closeEOMNI)

    def showEOMNI(self):
        if self.electronOMNI:
            self.electronOMNI.show()
            self.electronOMNI.update()

    def closeEOMNI(self):
        if self.electronOMNI:
            self.electronOMNI.close()
            self.electronOMNI = None

    def startEPAD(self):
        self.closeEPAD()
        if self.electronPAD is None or self.electronPAD.wasClosed:
            self.electronPAD = ElectronPitchAngle(self)
            self.initGeneralSelect('Electron PAD', '#0a22ff', self.electronPAD.ui.timeEdit, 
            'Single', self.showEPAD, closeFunc=self.closeEPAD)

    def editEPAD(self):
        self.endGeneralSelect()
        self.startEPAD() # Start up actual EPAD object
        kws = [self.electronPAD.lowKw, self.electronPAD.midKw, 
            self.electronPAD.hiKw]

        # Find matching color plots
        matchingPlts = []
        for kw in kws:
            kw = 'E' + kw
            for lbl in self.pltGrd.labels:
                if kw in lbl.dstrs:
                    pltIndex = self.pltGrd.labels.index(lbl)
                    matchingPlts.append(self.plotItems[pltIndex])

        if matchingPlts == []:
            return

        # Set log mode combo box
        basePlot = matchingPlts[0]
        colorPltIndex = self.pltGrd.colorPlts.index(basePlot)
        logMode = self.pltGrd.colorPltElems[colorPltIndex][0].logMode
        if not logMode:
            self.electronPAD.ui.scaleModeBox.setCurrentIndex(1)

        # Get value ranges from each gradient and toggle/set in EPAD plot windw
        indexDict = { 'E'+kws[0]:2, 'E'+kws[1]:1, 'E'+kws[2]:0 }
        for plt in matchingPlts:
            colorPltIndex = self.pltGrd.colorPlts.index(plt)
            name = self.pltGrd.colorPltNames[colorPltIndex]
            grad, lbl = self.pltGrd.colorPltElems[colorPltIndex]
            minVal, maxVal = grad.valueRange
            layoutIndex = indexDict[name] # EPAD window index

            self.electronPAD.ui.valRngSelectToggled(layoutIndex, True)
            self.electronPAD.ui.rangeToggles[layoutIndex].setChecked(True)
            minBox, maxBox = self.electronPAD.ui.rangeElems[layoutIndex][0:2]
            minBox.setValue(minVal)
            maxBox.setValue(maxVal)

        # Auto-select first plot's time range
        plotTimes = basePlot.listDataItems()[1].times
        self.currSelect.leftClick(plotTimes[0], 0)
        self.currSelect.leftClick(plotTimes[-1], 0)

    def editEOmniPlots(self):
        self.endGeneralSelect()
        self.startEOMNI() # Start up actual EPAD object

        # Find matching color plots
        matchingPlts = []
        for lbl in self.pltGrd.labels:
            if 'Electron Spectrum' in lbl.dstrs or 'Ion Spectrum' in lbl.dstrs:
                pltIndex = self.pltGrd.labels.index(lbl)
                matchingPlts.append(self.plotItems[pltIndex])

        if matchingPlts == []:
            return

        # Set log mode combo box
        basePlot = matchingPlts[0]
        colorPltIndex = self.pltGrd.colorPlts.index(basePlot)
        logMode = self.pltGrd.colorPltElems[colorPltIndex][0].logMode
        if not logMode:
            self.electronOMNI.ui.scaleBox.setCurrentIndex(1)

        # Get value ranges from each gradient and toggle/set in EPAD plot windw
        for plt in matchingPlts:
            colorPltIndex = self.pltGrd.colorPlts.index(plt)
            name = self.pltGrd.colorPltNames[colorPltIndex]
            grad, lbl = self.pltGrd.colorPltElems[colorPltIndex]
            minVal, maxVal = grad.valueRange
            if 'Electron' in name:
                kw = 'Electron'
            else:
                kw = 'Ion'

            self.electronOMNI.ui.valToggles[kw].setChecked(True)
            maxBox, minBox = self.electronOMNI.ui.valBoxes[kw]
            minBox.setValue(minVal)
            maxBox.setValue(maxVal)

        # Auto-select first plot's time range
        plotTimes = basePlot.listDataItems()[1].times
        self.currSelect.leftClick(plotTimes[0], 0)
        self.currSelect.leftClick(plotTimes[-1], 0)

    def closeEPAD(self):
        if self.electronPAD:
            self.electronPAD.close()
            self.electronPAD = None
    
    def showEPAD(self):
        if self.electronPAD:
            self.clearStatusMsg()
            self.electronPAD.show()
            self.electronPAD.update()

    def openCurlometer(self):
        self.closeMMSTools()
        self.curlometer = Curlometer(self)
        self.initGeneralSelect('Curlometer', '#ffa500', self.curlometer.ui.timeEdit, 
            'Line', self.showCurlometer, self.updateCurlometer, 
            closeFunc=self.closeCurlometer)

    def showCurlometer(self):
        if self.curlometer:
            self.curlometer.show()
            self.curlometer.calculate()

    def closeCurlometer(self):
        if self.curlometer:
            self.curlometer.close()
            self.curlometer = None

    def openCurvature(self):
        self.closeMMSTools()
        self.curvature = Curvature(self)
        self.initGeneralSelect('Curvature', '#ff4242', self.curvature.ui.timeEdit,
            'Line', self.showCurvature, self.updateCurvature, self.closeCurvature)

    def showCurvature(self):
        if self.curvature:
            self.curvature.show()
            self.curvature.updateCalculations()

    def closeCurvature(self):
        if self.curvature:
            self.endGeneralSelect()
            self.curvature.close()

    def openPlaneNormal(self):
        self.closeMMSTools()
        self.planeNormal = PlaneNormal(self)
        self.initGeneralSelect('Plane Normal', '#42f495', None, 'Single',
            self.showNormal, closeFunc=self.closePlaneNormal)

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

    def openTraceStats(self):
        self.closePlotTools()
        self.traceStats = TraceStats(self)
        self.traceStats.show()

    def startDynamicSpectra(self):
        self.closePlotTools()
        if not self.dynSpectra or self.dynSpectra.wasClosed:
            self.dynSpectra = DynamicSpectra(self)
            self.initGeneralSelect('Dynamic Spectra', '#c700ff', self.dynSpectra.ui.timeEdit,
                'Single', self.showDynamicSpectra, self.updateDynamicSpectra,
                closeFunc=self.closeDynamicSpectra)
            self.showStatusMsg('Selecting dynamic spectrogram range...')

    def showDynamicSpectra(self):
        if self.dynSpectra:
            self.clearStatusMsg()
            self.dynSpectra.show()
            self.dynSpectra.updateParameters()
            self.dynSpectra.update()

    def closeDynamicSpectra(self):
        if self.dynSpectra:
            self.clearStatusMsg()
            self.dynSpectra.close()
            self.dynSpectra = None

    def startDynamicCohPha(self):
        self.closePlotTools()
        if not self.dynCohPha or self.dynCohPha.wasClosed:
            self.dynCohPha = DynamicCohPha(self)
            self.showStatusMsg('Selecting dynamic coherence/phase range...')
            self.initGeneralSelect('Dynamic Coh/Pha', '#c551ff', self.dynCohPha.ui.timeEdit,
                'Single', self.showDynamicCohPha, self.updateDynCohPha, 
                closeFunc=self.closeDynamicCohPha)

    def showDynamicCohPha(self):
        if self.dynCohPha:
            self.clearStatusMsg()
            self.dynCohPha.show()
            self.dynCohPha.updateParameters()
            self.dynCohPha.update()

    def closeDynamicCohPha(self):
        if self.dynCohPha:
            self.clearStatusMsg()
            self.dynCohPha.close()
            self.dynCohPha = None

    def startSpectra(self):
        self.closePlotTools()
        if not self.spectra or self.spectra.wasClosed:
            self.spectra = Spectra(self)
            self.showStatusMsg('Selecting spectra range...')
            self.initGeneralSelect('Spectra', '#c551ff', self.spectra.ui.timeEdit,
                'Single', self.showSpectra, closeFunc=self.closeSpectra)

    def showSpectra(self):
        if self.spectra:
            self.clearStatusMsg()
            self.spectra.show()
            self.spectra.initPlots()
            PyQtUtils.moveToFront(self.spectra)

    def startDynWave(self):
        self.closeDynWave()
        if not self.dynWave or self.dynWave.wasClosed:
            self.dynWave = DynamicWave(self)
            self.initGeneralSelect('Wave Analysis', None, self.dynWave.ui.timeEdit,
                'Single', self.showDynWave, self.updateDynWave,
                closeFunc=self.closeDynWave)

    def showDynWave(self):
        if self.dynWave:
            self.dynWave.show()
            self.dynWave.setUserSelections()
            self.dynWave.update()

    def closeDynWave(self):
        if self.dynWave:
            self.dynWave.close()
            self.dynWave = None

    def startDetrend(self):
        self.closePlotTools()
        self.detrendWin = DetrendWindow(self)
        self.showStatusMsg('Selecting region of data to detrend...')
        self.initGeneralSelect('Detrend', '#00d122', self.detrendWin.ui.timeEdit,
            'Single', self.showDetrend, closeFunc=self.closeDetrend)

    def showDetrend(self):
        if self.detrendWin:
            self.clearStatusMsg()
            self.detrendWin.plotDetrendDta()
            self.detrendWin.show()

    def closeDetrend(self):
        if self.detrendWin:
            self.detrendWin.close()
            self.detrendWin = None

    def startSmoothing(self):
        self.closeSmoothing()
        if self.edit:
            self.edit.showMinimized()
            self.smoothing = SmoothingTool(self, self.edit)
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
        # add some lines used to show where time series sliders will zoom to
        # trackerLine = pg.InfiniteLine(movable=False, angle=90, pos=0, pen=self.trackerPen)
        # pi.addItem(trackerLine)
        # self.trackerLines.append(trackerLine)

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
        self.newVars.append(dstr)
        self.DATASTRINGS.append(dstr)
        self.ABBRV_DSTR_DICT[dstr] = dstr

        # Use any datastring's times as base
        if times is None:
            times = self.getTimes(self.DATASTRINGS[0], 0)
        self.TIMES.append(times)
        self.TIMEINDEX[dstr] = len(self.TIMES) - 1

        # Add in data to dictionaries, no units
        self.ORIGDATADICT[dstr] = dta
        self.DATADICT[dstr] = [dta]
        self.UNITDICT[dstr] = units

        # Pad rest of datadict to have same length
        length = len(self.editHistory)
        while len(self.DATADICT[dstr]) < length:
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
        self.errorFlag = 1e16 # overriding for now since the above line is sometimes wrong depending on the file (i think bx saves as 1e31 but doesnt update header)
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

    def updateXRange(self):
        # Update bottom axis' time label
        rng = self.getSelectedTimeRange()
        timeLbl = self.getTimeLabel(rng)
        self.pltGrd.setTimeLabel(timeLbl)

        for pi in self.plotItems:
            pi.setXRange(self.tO-self.tickOffset, self.tE-self.tickOffset, 0.0)

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
            mmsColors = ['000005', 'd55e00', '009e73', '56b4e9']
            for dstrLst in dstrs:
                penLst = []
                for (currDstr, en), color in zip(dstrLst, mmsColors):
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
        dstrs,links = self.getDefaultPlotInfo()

        self.plotData(dstrs, links)

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

    def plotData(self, dataStrings, links):
        self.closeTraceStats() # Clear any regions
        self.endGeneralSelect()

        # save what the last plotted strings and links are for other modules
        self.lastPlotStrings = dataStrings
        self.lastPlotLinks = links

        self.plotItems = []
        self.labelItems = []

        # a list of pens for each trace (saved for consistency with spectra)
        self.plotTracePens = []

        # Store any previous label sets (for current file)
        prevLabelSets = []

        # Store old plot grid info in case color plts are replotted
        oldPltGrd = None
        if self.pltGrd is not None:
            oldPltGrd = self.pltGrd
            for plt in oldPltGrd.colorPlts:
                oldPltGrd.removeItem(plt)
                for trackerLine in self.trackerLines: # Remove old tracker
                    if trackerLine in plt.items:
                        plt.removeItem(trackerLine)
                        trackerLine.deleteLater()
            for cb, cl in oldPltGrd.colorPltElems:
                if cb is not None:
                    oldPltGrd.removeItem(cb)
                if cl is not None:
                    oldPltGrd.removeItem(cl)

        # Clear previous grid
        self.ui.glw.clear()

        # Add label for file name at top right
        fileNameLabel = BLabelItem()
        fileNameLabel.opts['justify'] = 'right'
        maxLabelWidth = BaseLayout.getMaxLabelWidth(fileNameLabel, self.ui.glw)
        fileNameLabel.setHtml(f"<span style='font-size:10pt;'>{self.getFileNameString(maxLabelWidth)}</span>")
        self.ui.glw.addItem(fileNameLabel, 0, 0, 1, 1)

        # Create new plot grid
        self.pltGrd = MainPlotGrid(self)
        self.ui.glw.addItem(self.pltGrd, 1, 0, 1, 1)

        self.trackerLines = []

        for plotIndex, dstrs in enumerate(dataStrings):
            # Check if special plot
            colorPlt = False
            for dstr, en in dstrs:
                if oldPltGrd and dstr in oldPltGrd.colorPltNames:
                    colorPlt = True
            if colorPlt:
                index = oldPltGrd.colorPltNames.index(dstr)
                plt = oldPltGrd.colorPlts[index]
                colorBar, gradLbl = oldPltGrd.colorPltElems[index]
                if gradLbl:
                    gradLbl.offsets = (2, 2)
                self.plotItems.append(plt)
                self.pltGrd.addColorPlt(plt, dstr, colorBar, gradLbl, oldPltGrd.colorPltUnits[index])
                self.plotTracePens.append([None])
                continue

            axis = DateAxis(self.epoch, orientation='bottom')
            topAxis = DateAxis(self.epoch, orientation='top')
            topAxis.setStyle(showValues=False)
            vb = MagPyViewBox(self, plotIndex)
            pi = MagPyPlotItem(viewBox = vb, axisItems={'bottom': axis, 
                'left': LinkedAxis(orientation='left'), 'top': topAxis })
            #pi.setClipToView(True) # sometimes cuts off part of plot so kinda trash?
            vb.enableAutoRange(x=False, y=False) # range is being set manually in both directions
            pi.setDownsampling(ds=1, auto=True, mode='peak')

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

                self.plotTrace(pi, dstr, editNum, pen)
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

        # Add in all previous label sets, if there are any
        for labelSetDstr in prevLabelSets:
            self.pltGrd.addLabelSet(labelSetDstr)

        self.updateXRange()
        self.updateYRange()

        for pi in self.plotItems:
            for item in pi.items:
                item.viewRangeChanged()

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
            if pi in self.pltGrd.colorPlts:
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
                subPltStrs.append((dstr, editNum))

                j+=1
            newPltStrs.append(subPltStrs)
        self.lastPlotStrings = newPltStrs
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

    def updateTraceStats(self):
        if self.traceStats:
            self.traceStats.update()

    def updateCurlometer(self):
        if self.curlometer:
            self.curlometer.calculate()

    def updateDynCohPha(self):
        if self.dynCohPha:
            self.dynCohPha.updateParameters()

    def updateDynamicSpectra(self):
        if self.dynSpectra:
            self.dynSpectra.updateParameters()

    def updateDynWave(self):
        if self.dynWave:
            self.dynWave.updateParameters()

    def updateCurvature(self):
        if self.curvature:
            self.curvature.updateCalculations()

    # color is hex string ie: '#ff0000' for red
    def initGeneralSelect(self, name, color, timeEdit, mode, startFunc, updtFunc=None, 
        closeFunc=None, canHide=False, maxSteps=1):
        self.endGeneralSelect()

        if timeEdit is not None:
            timeEdit.linesConnected = False
        self.currSelect = GeneralSelect(self, mode, name, color, timeEdit,
            func=startFunc, updtFunc=updtFunc, closeFunc=closeFunc, maxSteps=maxSteps)

    def endGeneralSelect(self):
        if self.currSelect:
            self.currSelect.closeAllRegions()
            self.currSelect = None

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
        assert(i0 <= i1)
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

    def autoSelectRange(self):
        # Automatically select the section currently being viewed
        t0, t1 = self.tO, self.tE
        region = LinkedRegion(self, self.plotItems, values=(t0, t1), 
            mode=self.selectMode, color=self.selectColor)
        self.regions.append(region)



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
        for a in [xAction, mouseAction]:
            self.menu.removeAction(a)

    def onLeftClick(self, ev):
        # map the mouse click to data coordinates
        mc = self.mapToView(ev.pos())
        x = mc.x()
        y = mc.y()

        if self.window.currSelect == None:
            self.window.openTraceStats()
            self.window.initGeneralSelect('Stats', None, self.window.traceStats.ui.timeEdit,
                'Adjusting', None, self.window.updateTraceStats, 
                closeFunc=self.window.closeTraceStats, maxSteps=-1)

        ctrlPressed = (ev.modifiers() == QtCore.Qt.ControlModifier)
        self.window.currSelect.leftClick(x, self.plotIndex, ctrlPressed)

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
        if self.window.currSelect:
            self.window.currSelect.rightClick(self.plotIndex)
        else:
            pg.ViewBox.mouseClickEvent(self, ev)

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