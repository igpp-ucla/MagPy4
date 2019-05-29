from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from MagPy4UI import TimeEdit, GridGraphicsLayout, PlotGrid, StackedLabel
from FF_Time import FFTIME
from dataDisplay import UTCQDate
from layoutTools import BaseLayout
from timeManager import TimeManager
from selectionManager import SelectableViewBox, GeneralSelect
from pyqtgraphExtensions import LinkedAxis, DateAxis
from traceStats import TraceStats

from dynamicSpectra import DynamicSpectra, DynamicCohPha
from spectra import Spectra

import pyqtgraph as pg
import numpy as np

from scipy import signal
import functools

class DetrendWindowUI(BaseLayout):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Detrend')
        Frame.resize(1100, 700)
        layout = QtWidgets.QGridLayout(Frame)

        # Set up grid graphics layout
        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout(window)
        self.gview.setCentralItem(self.glw)
        self.glw.layout.setHorizontalSpacing(10)
        self.glw.layout.setContentsMargins(15, 10, 25, 10)
        layout.addWidget(self.gview, 1, 0, 1, 1)

        # Set up top settings/tools area
        self.btns = []
        toolsFrame = self.setupToolsLayout()

        settingsFrame = self.setupSettingsLayout()

        topLayout = QtWidgets.QHBoxLayout()
        topLayout.addWidget(toolsFrame)
        topLayout.addWidget(settingsFrame)
        topLayout.addStretch()
        layout.addLayout(topLayout, 0, 0, 1, 1)

        # Set up time edits at bottom of window
        timeLt, self.timeEdit, self.statusBar = self.getTimeStatusBar()
        layout.addLayout(timeLt, 2, 0, 1, 1)

    def setupSettingsLayout(self):
        frame = QtWidgets.QGroupBox('Settings')
        layout = QtWidgets.QHBoxLayout(frame)

        self.linkPlotsChk = QtWidgets.QCheckBox('Link Plots ')
        self.linkPlotsChk.setChecked(True)

        self.updtBtn = QtWidgets.QPushButton('Update')

        for e in [self.linkPlotsChk, None, self.updtBtn, None, None]:
            if e:
                layout.addWidget(e)
            else:
                spacer = self.getSpacer(5)
                layout.addItem(spacer)

        return frame

    def setupToolsLayout(self):
        frame = QtWidgets.QGroupBox('Tools')
        layout = QtWidgets.QHBoxLayout(frame)
        for tool in ['Spectra', 'Dynamic Spectra', 'Dynamic Coh/Pha']:
            btn = QtWidgets.QPushButton(tool)
            layout.addWidget(btn)
            self.btns.append(btn)
        return frame

class DetrendWindow(QtGui.QFrame, DetrendWindowUI, TimeManager):
    # Plots detrended data and mimics some of main window's functionality
    def __init__(self, window, parent=None):
        super(DetrendWindow, self).__init__(parent)
        self.window = window
        self.ui = DetrendWindowUI()
        self.ui.setupUI(self, window)

        # Initialize time parameters
        minTime, maxTime = window.getSelectionStartEndTimes()
        TimeManager.__init__(self, minTime, maxTime, window.epoch)

        # Store other related state information
        self.OS = window.OS
        self.currentEdit = self.window.currentEdit
        self.resolution = self.window.resolution
        self.tickOffset = self.window.tickOffset
        self.pens = window.pens
        self.edit = None
        self.traceStatsOnTop = True

        # State modified upon plotting
        self.currSelect = None
        self.lastPlotStrings = []
        self.plotItems = []
        self.dtDatas = {}
        self.DATASTRINGS = []

        self.modifier = '_DT' # String appended to variable names

        # Analysis tools
        self.spectra = None
        self.dynSpectra = None
        self.dynCohPha = None
        self.traceStats = None

        # Connect buttons to functions
        btnFuncs = [self.startSpectra,self.startDynSpectra,self.startDynCohPha]
        for btn, btnFunc in zip(self.ui.btns, btnFuncs):
            btn.clicked.connect(btnFunc)

        self.ui.updtBtn.clicked.connect(self.plotDetrendDta)

    def closeEvent(self, event):
        self.closeSubWindows()
        self.window.endGeneralSelect()
        self.close()

    def endGeneralSelect(self):
        if self.currSelect:
            self.currSelect.closeAllRegions()
            self.currSelect = None

    def closeSubWindows(self):
        self.closeSpectra()
        self.closeDynSpectra()
        self.closeDynCohPha()
        self.closeTraceStats()

    def startSpectra(self):
        self.closeSubWindows()
        self.spectra = Spectra(self)
        self.currSelect = GeneralSelect(self, 'Single', 'Spectra', '#0000d1',
            self.spectra.ui.timeEdit, self.openSpectra, 
            closeFunc=self.closeSpectra)
        self.autoSelect()

    def openSpectra(self):
        if self.spectra:
            self.spectra.initPlots()
            self.spectra.show()

    def closeSpectra(self):
        if self.spectra:
            self.endGeneralSelect()
            self.spectra.close()
            self.spectra = None

    def startDynSpectra(self):
        self.closeSubWindows()
        self.dynSpectra = DynamicSpectra(self)
        self.currSelect = GeneralSelect(self, 'Single', 'Dynamic Spectra', 
            '#cc0000', self.dynSpectra.ui.timeEdit, self.openDynSpectra, 
            self.updateDynSpectra, closeFunc=self.closeDynSpectra)
        self.autoSelect()

    def openDynSpectra(self):
        if self.dynSpectra:
            self.dynSpectra.updateParameters()
            self.dynSpectra.update()
            self.dynSpectra.show()

    def closeDynSpectra(self):
        if self.dynSpectra:
            self.endGeneralSelect()
            self.dynSpectra.close()
            self.dynSpectra = None

    def startDynCohPha(self):
        self.closeSubWindows()
        self.dynCohPha = DynamicCohPha(self)
        self.currSelect = GeneralSelect(self, 'Single', 'Dynamic Coh/Pha', 
            '#00d643', self.dynCohPha.ui.timeEdit, self.openDynCohPha, 
            self.updateDynCohPha, closeFunc=self.closeDynCohPha)
        self.autoSelect()

    def openDynCohPha(self):
        if self.dynCohPha:
            self.dynCohPha.updateParameters()
            self.dynCohPha.update()
            self.dynCohPha.show()

    def closeDynCohPha(self):
        if self.dynCohPha:
            self.endGeneralSelect()
            self.dynCohPha.close()
            self.dynCohPha = None

    def openTraceStats(self):
        self.closeSubWindows()
        self.traceStats = TraceStats(self)

        # Remove buttons unnecessary for detrend window
        viewBtn = self.traceStats.ui.dispRangeBtn
        dtaBtn = self.traceStats.ui.dtaBtn
        for elem in [viewBtn, dtaBtn]:
            self.traceStats.ui.layout.removeWidget(elem)
            elem.deleteLater()

        self.traceStats.show()

    def closeTraceStats(self):
        if self.traceStats:
            self.endGeneralSelect()
            self.traceStats.close()
            self.traceStats = None

    def updateTraceStats(self):
        if self.traceStats:
            self.traceStats.update()

    def updateDynCohPha(self):
        if self.dynCohPha:
            self.dynCohPha.updateParameters()

    def updateDynSpectra(self):
        if self.dynSpectra:
            self.dynSpectra.updateParameters()

    def autoSelect(self):
        # Auto-selects the entire range of detrended data
        if self.currSelect:
            self.currSelect.leftClick(self.tO-self.tickOffset, 0)
            self.currSelect.leftClick(self.tE-self.tickOffset, 0)

    def plotDetrendDta(self):
        # Clear previous state
        self.closeSubWindows()
        self.lastPlotStrings = []
        self.plotItems = []
        self.dtDatas = {}

        # Set up grid elements
        self.ui.glw.clear()
        self.pltGrd = PlotGrid(self)
        self.ui.glw.addItem(self.pltGrd)
        self.pltGrd.addItem(pg.LabelItem('Detrended Data'), 0, 1, 1, 1)

        plotNum = 0
        for pltInfo in self.window.getSelectedPlotInfo():
            # Extract variable names, edit nums, and pens from plot info
            plotStrings, pens = pltInfo
            dstrs, ens = [], []
            for dstr, en in plotStrings:
                if dstr == '':
                    continue
                dstrs.append(dstr)
                ens.append(en)

            # Build plot item
            la = LinkedAxis(orientation='left')
            ba = DateAxis(orientation='bottom')
            vb = SelectableViewBox(self, plotNum)
            plt = pg.PlotItem(viewBox=vb, axisItems={'left':la, 'bottom':ba})
            self.plotItems.append(plt)

            # Build plot label
            colors = [pen.color().name() for pen in pens]
            stackLbl = self.window.buildStackedLabel(plotStrings, colors)
            if stackLbl.units:
                modifiedLabels = stackLbl.dstrs.copy()[:-1]
            else:
                modifiedLabels = stackLbl.dstrs.copy()
            modifiedLabels[0] += self.modifier
            stackLbl = StackedLabel(modifiedLabels, stackLbl.colors, stackLbl.units)

            # Add to grid
            self.pltGrd.addPlt(plt, stackLbl)
            pltStrs = []
            for dstr, pen in zip(dstrs, pens):
                modifiedStr = dstr + self.modifier
                startIndex, endIndex = self.window.calcDataIndicesFromLines(dstr, 
                    self.window.currentEdit)

                # Get subset of data/times currently selected
                dta = self.window.getData(dstr, self.window.currentEdit)
                dtaSubset = dta[startIndex:endIndex]
                times = self.window.getTimes(dstr, self.window.currentEdit)[0]
                times = times - self.tickOffset
                timeSubset = times[startIndex:endIndex]

                # Detrend the data and store it in dictionary
                detrendType = 'linear'
                dtData = signal.detrend(dtaSubset, type=detrendType)
                self.dtDatas[modifiedStr] = dtData

                # Plot and store for lastPlotStrings
                plt.plot(timeSubset, dtData, pen=pen)
                pltStrs.append((modifiedStr, 0))

            # Update time range and axis appearance settings
            plt.setXRange(timeSubset[0], timeSubset[-1], 0.0)
            self.tO = timeSubset[0] + self.tickOffset
            self.tE = timeSubset[-1] + self.tickOffset
            self.minTime = self.tO
            self.maxTime = self.tE
            self.adjustPlotAppr(plt)

            self.lastPlotStrings.append(pltStrs)
            plotNum += 1
        self.ui.timeEdit.setupMinMax(self.getMinAndMaxDateTime())
        self.DATASTRINGS = list(self.dtDatas.keys())
        self.ui.glw.update()

        # Set time label
        startTime, endTime = self.window.getSelectionStartEndTimes()
        rng = endTime - startTime
        mode = self.window.getTimeLabelMode(rng)
        lbl = self.window.getTimeLabel(rng)
        self.pltGrd.setTimeLabel(lbl)

        # Update time ticks
        ba.window = self.window
        ba.tickOffset = self.tickOffset
        ba.updateTicks(self.window, mode, timeRange=(startTime, endTime))
        ba.setStyle(showValues=True)

        # Link plots if necessary
        if self.ui.linkPlotsChk.isChecked():
            dtas = [i for k,i in self.dtDatas.items()]
            self.linkPlots(dtas, self.plotItems)

    def clearStatusMsg(self):
        return

    def linkPlots(self, dtas, plts):
        # Re-implementation of plot linking code in updateYRange of main window
        ranges = []
        maxDiff = 0
        for dta in dtas:
            minVal = np.min(dta)
            maxVal = np.max(dta)

            if np.isnan(minVal) or np.isnan(maxVal):
                return

            ranges.append((minVal, maxVal))
            maxDiff = max(maxDiff, maxVal-minVal)

        for plt, (minVal, maxVal) in zip(plts, ranges):
            l2 = (maxDiff - (maxVal - minVal)) / 2
            plt.setYRange(minVal - l2, maxVal+l2, padding=0.05)

    def adjustPlotAppr(self, plt):
        # Show all axes and hide all values except on left axis
        plt.showAxis('top')
        plt.showAxis('right')
        for ax in ['bottom', 'top', 'right']:
            plt.getAxis(ax).setStyle(showValues=False)
        plt.hideButtons()

    def getTimes(self, dstr, en):
        dstr = self.stripName(dstr)
        return self.window.getTimes(dstr, en)

    def getSelectionStartEndTimes(self, regNum=0):
        # Returns detrend window's selected regions
        if self.currSelect is None or self.currSelect.regions == []:
            return self.tO, self.tE
        t0, t1 = self.currSelect.regions[regNum].getRegion()
        return (t0,t1) if t0 <= t1 else (t1,t0)

    def getMinAndMaxDateTime(self):
        tO, tE = self.getSelectionStartEndTimes()
        minDateTime = UTCQDate.UTC2QDateTime(FFTIME(tO, Epoch=self.epoch).UTC)
        maxDateTime = UTCQDate.UTC2QDateTime(FFTIME(tE, Epoch=self.epoch).UTC)
        return minDateTime, maxDateTime

    def getData(self, dstr, en=None):
        # Get full dataset and fill the selected range with the detrended data
        # (for compatability purposes with analysis tools)
        if en is None:
            en = self.currentEdit
        dstr = self.stripName(dstr)

        a, b = self.window.calcDataIndicesFromLines(dstr, en)
        dta = self.window.getData(dstr, en)

        detrendDta = self.dtDatas[dstr+self.modifier]
        dta = dta.copy()
        dta[a:b] = detrendDta
        return dta

    def getPrunedData(self, dstr, en, a, b):
        dta = self.getData(dstr, en)[a:b]
        return dta[dta < self.window.errorFlag]

    def calcDataIndicesFromLines(self, dstr, editNumber, regNum=0):
        # Re-implemented from main window version to use detrend window's
        # functions that handle the modifier string in the dstr
        times = self.getTimes(dstr, editNumber)[0]
        t0, t1 = self.getSelectionStartEndTimes(regNum)
        i0 = self.calcDataIndexByTime(times, t0)
        i1 = self.calcDataIndexByTime(times, t1)
        if i1 > len(times)-1:
            i1 = len(times)-1
        assert(i0 <= i1)
        return i0,i1

    def getSelectedPlotInfo(self):
        # Adds the modifier to the plot info dstrs acquired from main window
        pltInfo = self.window.getSelectedPlotInfo()
        modifiedPltInfo = []
        for subPltInfo in pltInfo:
            subDstrs, pens = subPltInfo
            newPltStrLst = []
            for dstr, en in subDstrs:
                dstr = dstr + self.modifier
                newPltStrLst.append((dstr, en))
            modifiedPltInfo.append((newPltStrLst, pens))

        return modifiedPltInfo

    def getLabel(self, dstr, en):
        dstr = self.stripName(dstr)
        return self.window.getLabel(dstr, en) + self.modifier

    def getFileNameString(self, width):
        return self.window.getFileNameString(width)

    def stripName(self, dstr):
        return dstr.strip(self.modifier)

    def getAbbrvDstr(self, dstr):
        dstr = self.stripName(dstr)
        return self.window.getAbbrvDstr(dstr) + self.modifier

    def getDefaultPlotInfo(self):
        return self.window.getDefaultPlotInfo()