

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from FF_Time import FFTIME
from plotAppearance import PlotAppearance, SpectraPlotApp
from pyqtgraphExtensions import GridGraphicsLayout, LogAxis, BLabelItem, SpectraPlotItem, MagPyAxisItem, MagPyPlotItem
from dataDisplay import UTCQDate
from MagPy4UI import TimeEdit
from spectraUI import SpectraUI, SpectraViewBox
from layoutTools import BaseLayout
from waveAnalysis import WaveAnalysis
from dynBase import SpectraBase
import functools
import time
from mth import Mth

class Spectra(QtWidgets.QFrame, SpectraUI, SpectraBase):
    def __init__(self, window, parent=None):
        super(Spectra, self).__init__(parent)
        self.window = window
        self.ui = SpectraUI()
        self.ui.setupUI(self, window)
        
        self.ui.updateButton.clicked.connect(self.update)
        self.ui.bandWidthSpinBox.valueChanged.connect(self.update)
        self.ui.separateTracesCheckBox.stateChanged.connect(self.clearAndUpdate)
        self.ui.aspectLockedCheckBox.stateChanged.connect(self.setAspect)
        self.ui.waveAnalysisButton.clicked.connect(self.openWaveAnalysis)
        self.ui.logModeCheckBox.toggled.connect(self.updateScaling)
        self.ui.plotApprAction.triggered.connect(self.openPlotAppr)
        self.ui.unitRatioCheckbox.stateChanged.connect(self.squarePlots)

        self.plotItems = []
        self.sumPlots = []
        self.cohPhaPlots = []
        self.rowItems = []
        self.tracePenList = []
        self.wasClosed = False
        self.waveAnalysis = None
        self.plotAppr = None
        self.linearMode = False
        self.bTotalKw = '-bt'

    def closeWaveAnalysis(self):
        if self.waveAnalysis:
            self.waveAnalysis.close()
            self.waveAnalysis = None

    def openWaveAnalysis(self):
        self.closeWaveAnalysis()
        self.waveAnalysis = WaveAnalysis(self, self.window)
        self.waveAnalysis.show()

    def openPlotAppr(self, sig, plotItems=None):
        if plotItems == None:
            plotItems = self.plotItems
            self.plotAppr = SpectraPlotApp(self, plotItems)
        else:
            self.plotAppr = PlotAppearance(self, plotItems)
        self.plotAppr.show()

    def closePlotAppr(self):
        if self.plotAppr:
            self.plotAppr.close()
            self.plotAppr = None

    def updateDelayed(self):
        self.ui.bandWidthSpinBox.setReadOnly(True)
        QtCore.QTimer.singleShot(100, self.update)

    def closeEvent(self, event):
        self.closeWaveAnalysis()
        self.closePlotAppr()
        self.window.endGeneralSelect()
        self.wasClosed = True # setting self.window.spectra=None caused program to crash with no errors.. couldnt figure out why so switched to this

    # return start and stop indices of selected data
    def getIndices(self, dstr, en):
        if dstr not in self.indices:
            i0,i1 = self.window.calcDataIndicesFromLines(dstr, en)
            self.indices[dstr] = (i0,i1)
        return self.indices[dstr]

    def getPoints(self, dstr, en):
        i0,i1 = self.getIndices(dstr, en)
        return i1-i0

    def getBw(self):
        bw = self.ui.bandWidthSpinBox.value()
        return bw

    def getFreqs(self, dstr, en):
        N = self.getPoints(dstr, en)
        if N not in self.freqs:
            self.freqs[N] = self.calculateFreqList(self.getBw(), N)
        return self.freqs[N]

    def getfft(self, dstr, en):
        if dstr not in self.ffts:
            i0,i1 = self.getIndices(dstr, en)
            fft = SpectraBase.getfft(self, dstr, en, i0, i1)
            self.ffts[dstr] = fft
        return self.ffts[dstr]

    # Used to make the appearance of logAxis axes and linear axes more uniform
    def setAxisAppearance(self, lst):
        for v in lst:
            v.tickFont = QtGui.QFont()
            v.tickFont.setPixelSize(14)

    def squarePlots(self):
        # If unit ratio, set plots to be square and axes to have same scaling
        if self.ui.unitRatioCheckbox.isChecked():
            for plt in self.plotItems + self.sumPlots:
                plt.squarePlot = True
                plt.getViewBox().setAspectLocked(lock=True, ratio=1.0)
                # Update plot sizes
                plt.resizeEvent(None)
        else:
            # Unlock aspect ratio and reset sizes to fill expand w/ grid
            for plt in self.plotItems + self.sumPlots:
                plt.squarePlot = False
                plt.getViewBox().setAspectLocked(lock=False)
                plt.adjustSize()
        self.ui.grid.resizeEvent(None)
        self.setAspect()

    def updateScaling(self, val=None):
        # TODO: Reset tick diff after updating scaling mode
        # Set the mode parameter
        if val is None:
            val = self.ui.logModeCheckBox.isChecked()
        self.linearMode = not val

        # Disable/Enable auto-range and set scaling mode for spectra plots
        for pi in self.plotItems:
            pi.enableAutoRange(x=True, y=False) # disable y auto scaling

            # Set up range/scaling modes
            if self.linearMode:
                pi.setLogMode(False, True)
            else:
                pi.setLogMode(True, True)
    
        # Manually set y-scale for each row in spectra plots
        for row in self.rowItems:
            self.setYRangeForRow(row)

        # Set log mode for coh/pha plots
        for pi in self.cohPhaPlots:
            if self.linearMode:
                pi.setLogMode(False, False)
            else:
                pi.setLogMode(True, False)

        # Set log mode for sum of power spectra plots
        for pi in self.sumPlots:
            if self.linearMode:
                pi.setLogMode(False, True)
            else:
                pi.setLogMode(True, True)

        # Set the bottom axis labels for all plots
        btmLabel = 'Log Frequency (Hz)'
        if self.linearMode:
            btmLabel = 'Frequency (Hz)'

        for plt in self.sumPlots + self.cohPhaPlots + self.plotItems:
            plt.getAxis('bottom').setLabel(btmLabel)

        # Reset aspect ratios
        self.setAspect(False)

        # Adjust aspect ratio settings w/ delay in case plots haven't been shown yet
        QtCore.QTimer.singleShot(100, self.setAspect)

    def setAspect(self, val=None):
        if val is None:
            val = self.ui.aspectLockedCheckBox.isChecked()
        if val:
            # Setting ratio to None locks in current aspect ratio
            for pi in self.plotItems + self.sumPlots:
                ratio = None
                pi.setAspectLocked(True, ratio=ratio)
        else: # Unlock aspect ratio for each plot and update graphs
            for pi in self.plotItems + self.sumPlots:
                pi.setAspectLocked(False)

    def updateCalculations(self):
        plotInfos = self.window.getSelectedPlotInfo()

        self.indices = {}
        self.freqs = {}
        self.ffts = {}
        self.powers = {}
        self.maxN = 0

        for li, (strList, penList) in enumerate(plotInfos):
            for i,(dstr,en) in enumerate(strList):
                fft = self.getfft(dstr,en)
                N = self.getPoints(dstr,en)
                self.maxN = max(self.maxN,N)
                power = self.calculatePower(self.getBw(), fft, N)
                self.powers[dstr] = power

        # calculate coherence and phase from pairs
        c0 = self.ui.cohPair0.currentText()
        c1 = self.ui.cohPair1.currentText()
        coh,pha = self.calculateCoherenceAndPhase(self.getBw(), self.getfft(c0,0), self.getfft(c1,0), self.getPoints(c0,0))
        self.coh = coh
        self.pha = pha

        # Additional updates for sum of powers
        if self.ui.combinedFrame.isChecked():
            vecDstrs = self.getVecDstrs()
            self.powers[self.bTotalKw] = self.calcMagPower(vecDstrs)

    # Initialize all plots that will be used with the current scaling
    # Sets up initial plots, label formats, axis types, and plot placement
    def initPlots(self, plotStrings, plotPens):
        # Clear all current plots
        self.ui.grid.clear()
        self.ui.grid.setNumCols(4)
        self.ui.labelLayout.clear()
        self.plotItems = []
        self.cohPhaPlots = []
        self.tracePenList = []

        # Get updated information about plots/settings
        oneTracePerPlot = self.ui.separateTracesCheckBox.isChecked()
        aspectLocked = self.ui.aspectLockedCheckBox.isChecked()

        # Build a plot item for each sub-list of traces/pens
        for pltStrsLst, plotPensLst in zip(plotStrings, plotPens):
            pi = self.buildPlotItem()
            self.plotItems.append(pi)
            self.ui.grid.addItem(pi, rowspan=1, colspan=1)

            # Set title font size
            pi.titleLabel.setAttr('size', '12pt')

            # Update trace pen list
            plotPensLst = [pg.mkPen(pen) for pen in plotPensLst]
            self.tracePenList.append(plotPensLst)

        # Update plot titles
        self.updateTitleColors(self.tracePenList)

        # Find largest title width and use it to set the minimum width
        # for all columns so they all have the same width
        maxWidth = 0
        for pi in self.plotItems:
            piw = pi.titleLabel._sizeHint[0][0] + 60 # gestimating padding
            maxWidth = max(maxWidth, piw)

        for c in range(0, self.ui.grid.layout.columnCount()):
            self.ui.grid.layout.setColumnMinimumWidth(c, maxWidth)

        # Add info about time range / frequency bands
        leftLabel = BLabelItem({'justify':'left'})
        rightLabel = BLabelItem({'justify':'left'})
        leftLabel.setHtml('File:<br>Frequency Bands:<br>Time:')

        self.ui.labelLayout.addItem(leftLabel)
        self.ui.labelLayout.nextColumn()
        self.ui.labelLayout.addItem(rightLabel)
        self.updateInfoLabel()

        # Set row items to be used when updating range for each row
        self.rowItems = self.ui.grid.getRowItems()

        # Add plot appearance menu to context menu for each plot:
        for plt in self.plotItems:
            plt.getViewBox().menu.addAction(self.ui.plotApprAction)

    def updateInfoLabel(self):
        # Get max label width and use it to align the info for
        # the file, frequency bands, and start/end times
        lbl = self.ui.labelLayout.getItem(0, 2)
        maxLabelWidth = BaseLayout.getMaxLabelWidth(lbl, self.ui.grid)
        t0,t1 = self.window.getSelectionStartEndTimes()
        startDate = UTCQDate.removeDOY(FFTIME(t0, Epoch=self.window.epoch).UTC)
        endDate = UTCQDate.removeDOY(FFTIME(t1, Epoch=self.window.epoch).UTC)
        lbl.setHtml(f'{self.window.getFileNameString(maxLabelWidth)}<br>{self.maxN}<br>{startDate} to {endDate}')

    def clearAndUpdate(self):
        self.ui.grid.clear()
        self.ui.labelLayout.clear()
        self.update()

    # Updates all current plots currently being viewed (unless scaling mode changes)
    def update(self):
        # Re-calculate power spectra and coh/pha
        self.updateCalculations()

        # Get plot info and parameters
        plotInfos = self.window.getSelectedPlotInfo()
        oneTracePerPlot = self.ui.separateTracesCheckBox.isChecked()
        plotStrings, plotPens = self.splitPlotInfo(plotInfos, oneTracePerPlot)

        # If grid is empty (nothing plotted yet), initialize plot items
        if self.ui.grid.layout.count() < 1:
            self.initPlots(plotStrings, plotPens)

        plotPens = self.tracePenList
        plts = self.plotItems
        # Clear each plot and re-plot traces w/ updated power results
        for plotDstrs, plotPenList, plt in zip(plotStrings, plotPens, plts):
            plt.clear()
            for (dstr, en), pen in zip(plotDstrs, plotPenList):
                freq = self.getFreqs(dstr, en)
                power = self.powers[dstr]
                plt.plot(freq, power, pen=pen)

        self.updateTitleColors(plotPens)
        self.updateInfoLabel()

        # Update coherence and phase graphs
        self.updateCohPha()
        self.updateCombined()
        self.updateScaling()

    def buildPlotItem(self):
        ba = LogAxis(orientation='bottom')
        la = LogAxis(orientation='left')
        pi = SpectraPlotItem(viewBox = SpectraViewBox(), axisItems={'bottom':ba, 'left':la})
        pi.hideButtons() # hide autoscale button

        return pi

    def getPower(self, dstr, en):
        # Return power for a dstr, calculating it if its not in the dictionary
        if dstr not in self.powers:
            bw = self.getBw()
            fft = self.getfft(dstr, en)
            a, b = self.getIndices(dstr, en)
            self.powers[dstr] = self.calculatePower(bw, fft, b-a)
        return self.powers[dstr]

    def calcSumOfPowers(self, vecDstrs):
        powerLst = []
        for dstr, en in vecDstrs:
            powerLst.append(np.array(self.getPower(dstr, en)))
        sumOfPowers = powerLst[0] + powerLst[1] + powerLst[2]
        return sumOfPowers

    def calcMagPower(self, vecDstrs):
        # Calculates the magnitude and then gets its power spectra
        bw = self.getBw()
        a, b = self.getIndices(vecDstrs[0][0], vecDstrs[0][1])
        dtaLst = [self.window.getData(dstr, en)[a:b] for dstr, en in vecDstrs]
        dtaLst = [dta ** 2 for dta in dtaLst]
        b_tot = np.sqrt(dtaLst[0] + dtaLst[1] + dtaLst[2])
        fft = fftpack.rfft(b_tot.tolist())
        power = self.calculatePower(bw, fft, b-a)
        return power

    def getVecDstrs(self):
        # Get list of dstrs from vector combo boxes
        dstrs = [box.currentText() for box in self.ui.axisBoxes]

        # Make a dictionary of the plotted dstrs and the maximum edit number shown
        enDict = {}
        plotInfo = self.window.lastPlotStrings
        for dstrLst in plotInfo:
            for dstr, en in dstrLst:
                if dstr in enDict:
                    enDict[dstr] = max(enDict[dstr], en)
                else:
                    enDict[dstr] = en

        # Return a list of tuples of the selected dstrs and their edit numbers
        vecDstrs = []
        for dstr in dstrs:
            if dstr in enDict:
                vecDstrs.append((dstr, enDict[dstr]))
            else:
                vecDstrs.append((dstr, self.window.currentEdit))

        return vecDstrs

    def updateCombined(self):
        if not self.ui.combinedFrame.isChecked():
            return
        elif not self.ui.tabs.isTabEnabled(3):
            self.ui.tabs.setTabEnabled(3, True)

        # Reset grid
        self.ui.sumGrid.clear()

        # Get default plot info
        self.sumPlots = []
        pltNames = ['Px + Py + Pz', 'Pt', '|Px + Py + Pz - Pt|']
        pens = [pg.mkPen(pen) for pen in self.window.pens]

        # Calculate/fetch sum of powers and related data
        vecDstrs = self.getVecDstrs()
        freq = self.getFreqs(vecDstrs[0][0], self.window.currentEdit)
        sumOfPowers = self.calcSumOfPowers(vecDstrs)
        magPower = self.powers[self.bTotalKw]
        absSumMinusPt = abs(sumOfPowers - magPower)

        # Create plots and set titles for plots w/ only one trace
        dta = [sumOfPowers, magPower, absSumMinusPt]
        coords = [(0, 0), (0, 1), (1, 0)]
        for dta, (x, y), pen, title in zip(dta, coords, pens[0:3], pltNames[0:3]):
            pi = self.buildPlotItem()
            self.ui.sumGrid.addItem(pi, x, y)
            self.sumPlots.append(pi)
            pi.plot(freq, dta, pen=pen)
            title = f"<span style='color:{pen.color().name()};'>{title}</span>"
            pi.titleLabel.setAttr('size', '12pt')
            pi.setTitle(title)

        # Build bottom right plot with |Px + Py + Pz - Pt| & Pt traces
        pi = self.buildPlotItem()
        self.ui.sumGrid.addItem(pi, 1, 1, 1, 1)
        pi.plot(freq, absSumMinusPt, pen=pens[2])
        pi.plot(freq, magPower, pen=pens[1])

        # Custom title string
        titleString = self.formatPltTitle(pltNames[2], pens[2].color().name())
        titleString = f"{titleString} & "
        titleString = f"{titleString}{self.formatPltTitle(pltNames[1], pens[1].color().name())}"
        pi.titleLabel.setAttr('size', '12pt')

        pi.setTitle(titleString)
        self.sumPlots.append(pi)

        # Set axis labels for all plots
        for pi in self.sumPlots:
            pi.setLabels(left='Power (nT<sup>2</sup> Hz<sup>-1</sup>)')

    def splitPlotInfo(self, plotInfos, sepTraceMode=False):
        # Determine which plot strings correspond to which plots
        # based on whether plotting separate traces or not
        plotStrings = []
        plotPens = []
        for dstrList, plotPenList in plotInfos:
            if sepTraceMode:
                for dstrInfo, penInfo in zip(dstrList, plotPenList):
                    plotStrings.append([dstrInfo])
                    plotPens.append([penInfo])
            else:
                plotStrings.append(dstrList)
                plotPens.append(plotPenList)

        return plotStrings, plotPens

    def updateTitleColors(self, penList):
        # Update title colors to match colors from a list of pens
        plotInfos = self.window.getSelectedPlotInfo()
        titleString = ''
        sepTraceMode = self.ui.separateTracesCheckBox.isChecked()

        # Determine which plot strings correspond to which plots
        # based on whether plotting separate traces or not
        plotStrings, prevPlotPens = self.splitPlotInfo(plotInfos, sepTraceMode)

        # Loop through each set of plot strings and the corresponding set of pens
        # in penList
        plotIndex = 0
        for stringInfo, plotPenList in zip(plotStrings, penList):
            pi = self.plotItems[plotIndex]
            titleString = ''
            # Update title string for every plot w/ new pen color
            for (dstr, en), pen in zip(stringInfo, plotPenList):
                pstr = self.window.getLabel(dstr, en)
                titleString = f"{titleString} <span style='color:{pen.color().name()};'>{pstr}</span>"
            pi.setTitle(titleString)
            plotIndex += 1

        self.tracePenList = penList

    def formatPltTitle(self, titleStr, color):
        return f"<span style='color:{color};'>{titleStr}</span>"

    # remove limits later incase they want to type in directly
    def removeLimits(self):
        self.ui.bandWidthSpinBox.setMinimum(1)
        self.ui.bandWidthSpinBox.setMaximum(99)
    
    def updateCohPha(self):
        c0 = self.ui.cohPair0.currentText()
        c1 = self.ui.cohPair1.currentText()
        abbrv0 = self.window.getAbbrvDstr(c0)
        abbrv1 = self.window.getAbbrvDstr(c1)
        freqs = self.getFreqs(c0,0)

        datas = [[self.ui.cohGrid, self.coh, 'Coherence', ''],[self.ui.phaGrid, self.pha, 'Phase', ' (&deg;)']]
        plts = []

        for d in datas:
            d[0].clear()
            ba = LogAxis(orientation='bottom')
            la = LogAxis(orientation='left')
            pi = MagPyPlotItem(axisItems={'bottom':ba, 'left':la})
            pi.plot(freqs, d[1], pen=QtGui.QPen(self.window.pens[0]))
            pi.titleLabel.setAttr('size', '12pt') # Set up default font size
            pi.setLabels(title=f'{d[2]}:  {abbrv0}   vs   {abbrv1}', left=f'{d[2]}{d[3]}')
            d[0].addItem(pi)
            plts.append(pi)

            # y axis should be in angles for phase
            if d[2] == 'Phase':
                la.setTickSpacing(90, 15)

        # Custom context menu handling for coh/pha
        actText = 'Change Plot Appearance...'
        self.plotApprActs = [QtWidgets.QAction(actText), QtWidgets.QAction(actText)]
        for pi, act in zip(plts, self.plotApprActs):
            act.triggered.connect(functools.partial(self.openPlotAppr, None, [pi]))
            vb = pi.getViewBox()
            vb.menu.addAction(act)

        self.cohPhaPlots = plts

    # scale each plot to use same y range
    # the viewRange function was returning incorrect results so had to do manually
    def setYRangeForRow(self, curRow):
        minVal = np.inf
        maxVal = -np.inf
        for item in curRow:
            dataItems = item.listDataItems()
            datas = [dataItem.yData for dataItem in dataItems]
            for dta in datas:
                minVal = min(minVal, min(dta))
                maxVal = max(maxVal, max(dta))
                                    
        minVal = np.log10(minVal) # since plots are in log mode have to give log version of range
        maxVal = np.log10(maxVal)
        for item in curRow:
            item.setYRange(minVal,maxVal)
