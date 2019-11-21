

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from FF_Time import FFTIME
from plotAppearance import PlotAppearance, SpectraPlotApp
from pyqtgraphExtensions import GridGraphicsLayout, LinearGraphicsLayout, LogAxis, BLabelItem, SpectraPlotItem, MagPyAxisItem, MagPyPlotItem
from dataDisplay import UTCQDate
from MagPy4UI import TimeEdit
from spectraUI import SpectraUI, SpectraViewBox
from layoutTools import BaseLayout
from waveAnalysis import WaveAnalysis
from dynamicSpectra import SpectraBase
import functools
import time
from mth import Mth

class Spectra(QtWidgets.QFrame, SpectraUI, SpectraBase):
    def __init__(self, window, parent=None):
        super(Spectra, self).__init__(parent)
        self.window = window
        self.ui = SpectraUI()
        self.ui.setupUI(self, window)
        
        self.ui.updateButton.clicked.connect(self.updateSpectra)
        self.ui.bandWidthSpinBox.valueChanged.connect(self.updateSpectra)
        self.ui.separateTracesCheckBox.stateChanged.connect(self.initPlots)
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
        QtCore.QTimer.singleShot(100, self.updateSpectra)

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
        QtCore.QTimer.singleShot(500, self.setAspect)

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
    def initPlots(self):
        # Clear all current plots
        self.ui.grid.clear()
        self.ui.labelLayout.clear()
        self.plotItems = []
        self.cohPhaPlots = []

        # Get updated information about plots/settings
        self.updateCalculations()
        plotInfos = self.window.getSelectedPlotInfo()
        oneTracePerPlot = self.ui.separateTracesCheckBox.isChecked()
        aspectLocked = self.ui.aspectLockedCheckBox.isChecked()

        numberPlots = 0
        rowItems = []
        curRow = []
        maxTitleWidth = 0
        self.tracePenList = []
        # For every plot in main window:
        for listIndex, (strList, penList) in enumerate(plotInfos):
            # Only copy color from main window
            penList = [pg.mkPen(pen.color()) for pen in penList]
            # For every trace/pen in the given plot:
            for i, (dstr, en) in enumerate(strList):
                # If first dstr or there are separate traces, create a new plot
                if i == 0 or oneTracePerPlot:
                    # Set up axes and viewBox
                    pi = self.buildPlotItem()

                    numberPlots += 1
                    titleString = ''
                    powers = []

                # Get x and y values
                freq = self.getFreqs(dstr,en)
                power = self.powers[dstr]
                powers.append(power)

                # Initialize pens, plot title, and plot data
                pen = penList[i]
                pstr = self.window.getLabel(dstr,en)
                titleString = f"{titleString} <span style='color:{pen.color().name()};'>{pstr}</span>"
                pi.titleLabel.setAttr('size', '12pt') # Set up default font size
                pi.plot(freq, power, pen=pen)

                # this part figures out layout of plots into rows depending on settings
                # also links the y scale of each row together
                lastPlotInList = i == len(strList) - 1
                if lastPlotInList or oneTracePerPlot:
                    pi.setLabels(title=titleString, left='Power (nT<sup>2</sup> Hz<sup>-1</sup>)')
                    piw = pi.titleLabel._sizeHint[0][0] + 60 # gestimating padding
                    maxTitleWidth = max(maxTitleWidth,piw)
                    self.ui.grid.addItem(pi)
                    self.plotItems.append(pi)
                    curRow.append((pi,powers))
                    if numberPlots % 4 == 0:
                        self.setYRangeForRow(curRow)
                        rowItems.append(curRow)
                        curRow = []
            self.tracePenList.append(penList)

        if curRow:
            self.setYRangeForRow(curRow)

        # otherwise gridlayout columns will shrink at different scales based on
        # the title string
        l = self.ui.grid.layout
        for i in range(l.columnCount()):
            l.setColumnMinimumWidth(i, maxTitleWidth)

        # add some text info like time range and file and stuff
        t0,t1 = self.window.getSelectionStartEndTimes()
        startDate = UTCQDate.removeDOY(FFTIME(t0, Epoch=self.window.epoch).UTC)
        endDate = UTCQDate.removeDOY(FFTIME(t1, Epoch=self.window.epoch).UTC)

        leftLabel = BLabelItem({'justify':'left'})
        rightLabel = BLabelItem({'justify':'left'})

        leftLabel.setHtml('File:<br>Frequency Bands:<br>Time:')
        maxLabelWidth = BaseLayout.getMaxLabelWidth(rightLabel, self.ui.grid)
        rightLabel.setHtml(f'{self.window.getFileNameString(maxLabelWidth)}<br>{self.maxN}<br>{startDate} to {endDate}')
           
        self.ui.labelLayout.addItem(leftLabel)
        self.ui.labelLayout.nextColumn()
        self.ui.labelLayout.addItem(rightLabel)

        self.rowItems = rowItems
        self.updateCohPha()
        self.updateCombined()

        self.updateScaling()

        # Add plot appearance menu to context menu for each plot:
        for plt in self.plotItems:
            plt.getViewBox().menu.addAction(self.ui.plotApprAction)

        # this is done to avoid it going twice with one click (ideally should make this multithreaded eventually so gui is more responsive)
        bw = self.ui.bandWidthSpinBox.value()
        self.ui.bandWidthSpinBox.setMinimum(max(1,bw-2))
        self.ui.bandWidthSpinBox.setMaximum(bw+2)
        QtCore.QTimer.singleShot(500, self.removeLimits)

    # Updates all current plots currently being viewed (unless scaling mode changes)
    def updateSpectra(self):
        self.updateCalculations()
        plotInfos = self.window.getSelectedPlotInfo()
        oneTracePerPlot = self.ui.separateTracesCheckBox.isChecked()
        plotNum = 0
        # For every plot spectra is generated from:
        for listIndex, (strList, penList) in enumerate(plotInfos):
            # Get the corresponding plot and clear it
            pi = self.plotItems[listIndex]
            if oneTracePerPlot == False: # Don't clear in this case bc listIndex != plotNum
                pi.clear()
            powers = []
            # For every trace in plot:
            for i, (dstr, en) in enumerate(strList):
                # Get current plot by plot number if separate traces are used
                if self.ui.separateTracesCheckBox.isChecked():
                    pi = self.plotItems[plotNum]
                    pi.clear()
                    powers = []
                    titleString = ''
                # Get x, y, and pen values and plot graph
                freq = self.getFreqs(dstr,en)
                power = self.powers[dstr]
                powers.append(power)
                pen = self.tracePenList[listIndex][i]
                pi.plot(freq, power, pen=pen)
                plotNum += 1

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

        # Reset grid and add spacer
        self.ui.sumGrid.clear()
        spacer = pg.LabelItem('')
        spacer.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum))
        self.ui.sumGrid.addItem(spacer, 2, 0, 1, 2)

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

        # Adjust aspect ratio settings w/ delay in case plots haven't been shown yet
        QtCore.QTimer.singleShot(100, self.setAspect)

    def updateTitleColors(self, penList):
        # Update title colors to match colors from a list of pens
        plotInfos = self.window.getSelectedPlotInfo()
        titleString = ''
        # For every plot
        for listIndex, (strList, prevPens) in enumerate(plotInfos):
            pi = self.plotItems[listIndex] # Get the corresponding plot
            # For every trace in plot:
            for i, (dstr, en) in enumerate(strList):
                if i == 0 or self.ui.separateTracesCheckBox.isChecked():
                    pi = self.plotItems[listIndex+i]
                    titleString = ''
                # Get new pen color from list corresponding to plot and trace num
                pen = penList[listIndex][i]
                pstr = self.window.getLabel(dstr,en)
                titleString = f"{titleString} <span style='color:{pen.color().name()};'>{pstr}</span>"
                pi.setTitle(titleString)

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
                angleTicks = [(x,f'{x}') for x in range(360,-361,-90)]
                la.setTicks([angleTicks])

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
        self.ui.grid.nextRow()
        minVal = np.inf
        maxVal = -np.inf
        for item in curRow:
            for pow in item[1]:
                minVal = min(minVal, min(pow))
                maxVal = max(maxVal, max(pow))
                                    
        #if np.isnan(minVal) or np.isinf(minVal) or np.isnan(maxVal) or
        #np.isinf(maxVal):
        minVal = np.log10(minVal) # since plots are in log mode have to give log version of range
        maxVal = np.log10(maxVal)
        for item in curRow:
            item[0].setYRange(minVal,maxVal)
