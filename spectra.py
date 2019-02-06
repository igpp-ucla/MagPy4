

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from FF_Time import FFTIME
from plotAppearance import PlotAppearance, SpectraPlotApp
from pyqtgraphExtensions import GridGraphicsLayout, LinearGraphicsLayout, LogAxis, BLabelItem
from dataDisplay import UTCQDate
from MagPy4UI import TimeEdit
from spectraUI import SpectraUI, SpectraViewBox
from waveAnalysis import WaveAnalysis
import functools
import time
from mth import Mth

class Spectra(QtWidgets.QFrame, SpectraUI):
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
        self.ui.logModeCheckBox.stateChanged.connect(self.updateScaling)
        self.ui.plotApprAction.triggered.connect(self.openPlotAppr)

        self.plotItems = []
        self.tracePenList = []
        self.window.setLinesVisible(False, 'general')
        self.wasClosed = False
        self.waveAnalysis = None
        self.plotAppr = None
        self.linearMode = False

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

    def getFreqs(self, dstr, en):
        N = self.getPoints(dstr, en)
        if N not in self.freqs:
            self.freqs[N] = self.calculateFreqList(N)
        return self.freqs[N]

    def getfft(self, dstr, en):
        if dstr not in self.ffts:
            i0,i1 = self.getIndices(dstr, en)
            data = self.window.getData(dstr, en)[i0:i1]
            fft = fftpack.rfft(data.tolist())
            self.ffts[dstr] = fft
        return self.ffts[dstr]

    # Used to make the appearance of logAxis axes and linear axes more uniform
    def setAxisAppearance(self, lst):
        for v in lst:
            v.tickFont = QtGui.QFont()
            v.tickFont.setPixelSize(14)

    def updateScaling(self):
        self.linearMode = not self.ui.logModeCheckBox.isChecked()
        self.initPlots()

    def setAspect(self):
        if self.ui.aspectLockedCheckBox.isChecked() == True:
            # Setting ratio to None locks in current aspect ratio
            for pi in self.plotItems:
                pi.setAspectLocked(True, ratio=None)
        else: # Unlock aspect ratio for each plot and update graphs
            for pi in self.plotItems:
                pi.setAspectLocked(False)
        self.updateSpectra()

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
                power = self.calculatePower(fft, N)
                self.powers[dstr] = power

        # calculate coherence and phase from pairs
        c0 = self.ui.cohPair0.currentText()
        c1 = self.ui.cohPair1.currentText()
        coh,pha = self.calculateCoherenceAndPhase(self.getfft(c0,0), self.getfft(c1,0), self.getPoints(c0,0))
        self.coh = coh
        self.pha = pha

    # Initialize all plots that will be used with the current scaling
    # Sets up initial plots, label formats, axis types, and plot placement
    def initPlots(self):
        # Clear all current plots
        self.ui.grid.clear()
        self.ui.labelLayout.clear()
        self.plotItems = []

        # Get updated information about plots/settings
        self.updateCalculations()
        plotInfos = self.window.getSelectedPlotInfo()
        oneTracePerPlot = self.ui.separateTracesCheckBox.isChecked()
        aspectLocked = self.ui.aspectLockedCheckBox.isChecked()

        numberPlots = 0
        curRow = []
        maxTitleWidth = 0
        # For every plot in main window:
        for listIndex, (strList, penList) in enumerate(plotInfos):
            # For every trace/pen in the given plot:
            for i, (dstr, en) in enumerate(strList):
                # If first dstr or there are separate traces, create a new plot
                if i == 0 or oneTracePerPlot:
                    # Set up axes and viewBox
                    ba = LogAxis(True,True,True,orientation='bottom')
                    la = LogAxis(True,True,True,orientation='left')
                    if self.linearMode:
                        ba = pg.AxisItem(orientation='bottom')
                        la = pg.AxisItem(orientation='left')
                        self.setAxisAppearance([ba,la])
                    pi = pg.PlotItem(viewBox = SpectraViewBox(), axisItems={'bottom':ba, 'left':la})

                    # Set up range/scaling modes
                    pi.setLogMode(True, True)
                    if self.linearMode:
                        pi.setLogMode(False, False)
                    pi.enableAutoRange(y=False) # disable y auto scaling so doesnt interfere with custom range settings
                    if self.linearMode:
                        pi.enableAutoRange(x=True, y=True)
                    pi.hideButtons() # hide autoscale button

                    numberPlots += 1
                    titleString = ''
                    powers = []

                # Get x and y values
                freq = self.getFreqs(dstr,en)
                power = self.powers[dstr]
                powers.append(power)

                # Initialize pens, plot title, and plot data
                pen = pg.mkPen(penList[i].color()) # Only copy color from main window
                pstr = self.window.getLabel(dstr,en)
                titleString = f"{titleString} <span style='color:{pen.color().name()};'>{pstr}</span>"
                pi.titleLabel.setAttr('size', '12pt') # Set up default font size
                pi.plot(freq, power, pen=pen)

                # this part figures out layout of plots into rows depending on settings
                # also links the y scale of each row together
                lastPlotInList = i == len(strList) - 1
                if lastPlotInList or oneTracePerPlot:
                    btmLabel = 'Log Frequency (Hz)'
                    if self.linearMode:
                        btmLabel = 'Frequency (Hz)'
                    pi.setLabels(title=titleString, left='Power (nT<sup>2</sup> Hz<sup>-1</sup>)', bottom=btmLabel)
                    piw = pi.titleLabel._sizeHint[0][0] + 60 # gestimating padding
                    maxTitleWidth = max(maxTitleWidth,piw)
                    self.ui.grid.addItem(pi)
                    self.plotItems.append(pi)
                    curRow.append((pi,powers))
                    if numberPlots % 4 == 0:
                        self.setYRangeForRow(curRow)
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
        maxLabelWidth = self.window.getMaxLabelWidth(rightLabel, self.ui.grid)
        rightLabel.setHtml(f'{self.window.getFileNameString(maxLabelWidth)}<br>{self.maxN}<br>{startDate} to {endDate}')
           
        self.ui.labelLayout.addItem(leftLabel)
        self.ui.labelLayout.nextColumn()
        self.ui.labelLayout.addItem(rightLabel)

        self.updateCohPha()

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

    # remove limits later incase they want to type in directly
    def removeLimits(self):
        self.ui.bandWidthSpinBox.setMinimum(1)
        self.ui.bandWidthSpinBox.setMaximum(99)
    
    def updateCohPha(self):
        c0 = self.ui.cohPair0.currentText()
        c1 = self.ui.cohPair1.currentText()
        abbrv0 = self.window.ABBRV_DSTR_DICT[c0]
        abbrv1 = self.window.ABBRV_DSTR_DICT[c1]
        freqs = self.getFreqs(c0,0)

        datas = [[self.ui.cohGrid, self.coh, 'Coherence', ''],[self.ui.phaGrid, self.pha, 'Phase', ' (&deg;)']]
        plts = []

        for d in datas:
            d[0].clear()
            ba = LogAxis(True,True,False,orientation='bottom')
            la = LogAxis(False,False,False,orientation='left')
            if self.linearMode:
                ba = pg.AxisItem(orientation='bottom')
                la = pg.AxisItem(orientation='left')
                self.setAxisAppearance([ba,la])
            pi = pg.PlotItem(axisItems={'bottom':ba, 'left':la})
            pi.setLogMode(True, False)
            if self.linearMode:
                pi.setLogMode(False, False)
            pi.plot(freqs, d[1], pen=QtGui.QPen(self.window.pens[0]))
            btmLabel = 'Log Frequency (Hz)'
            if self.linearMode:
                btmLabel = 'Frequency (Hz)'
            pi.titleLabel.setAttr('size', '12pt') # Set up default font size
            pi.setLabels(title=f'{d[2]}:  {abbrv0}   vs   {abbrv1}', left=f'{d[2]}{d[3]}', bottom=btmLabel)
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
        if not self.linearMode:
            minVal = np.log10(minVal) # since plots are in log mode have to give log version of range
            maxVal = np.log10(maxVal)
        for item in curRow:
            item[0].setYRange(minVal,maxVal)

    def getCommonVars(self, N):
        bw = self.ui.bandWidthSpinBox.value()
        if bw % 2 == 0: # make sure its odd 
            bw += 1
            self.ui.bandWidthSpinBox.setValue(bw)
        kmo = int((bw + 1) * 0.5)
        nband = (N - 1) / 2
        half = int(bw / 2)
        nfreq = int(nband - bw + 1)
        return bw,kmo,nband,half,nfreq

    def calculateFreqList(self, N):
        bw,kmo,nband,half,nfreq = self.getCommonVars(N)
        nfreq = int(nband - half + 1) #try to match power length
        C = N * self.window.resolution
        freq = np.arange(kmo, nfreq) / C
        #return np.log10(freq)
        if len(freq) < 2:
            print('Proposed spectra plot invalid!\nFrequency list has lass than 2 values')
            return None
        return freq

    def calculatePower(self, fft, N):
        bw,kmo,nband,half,nfreq = self.getCommonVars(N)
        C = 2 * self.window.resolution / N
        fsqr = [ft * ft for ft in fft]
        power = [0] * nfreq
        for i in range(nfreq):
            km = kmo + i
            kO = int(km - half)
            kE = int(km + half) + 1

            power[i] = sum(fsqr[kO * 2 - 1:kE * 2 - 1]) / bw * C

        return power

    def calculateCoherenceAndPhase(self, fft0, fft1, N):
        bw,kmo,nband,half,nfreq = self.getCommonVars(N)
        kStart = kmo - half
        kSpan = half * 4 + 1

        csA = fft0[:-1] * fft1[:-1] + fft0[1:] * fft1[1:]
        qsA = fft0[:-1] * fft1[1:] - fft1[:-1] * fft0[1:]
        pAA = fft0[:-1] * fft0[:-1] + fft0[1:] * fft0[1:]
        pBA = fft1[:-1] * fft1[:-1] + fft1[1:] * fft1[1:]

        csSum = np.zeros(nfreq)
        qsSum = np.zeros(nfreq)
        pASum = np.zeros(nfreq)
        pBSum = np.zeros(nfreq)

        for n in range(nfreq):
            KO = (kStart + n) * 2 - 1
            KE = KO + kSpan

            csSum[n] = sum(csA[KO:KE:2])
            qsSum[n] = sum(qsA[KO:KE:2])
            pASum[n] = sum(pAA[KO:KE:2])
            pBSum[n] = sum(pBA[KO:KE:2])

        coh = (csSum * csSum + qsSum * qsSum) / (pASum * pBSum)
        pha = np.arctan2(qsSum, csSum) * Mth.R2D

        # wrap phase
        n = pha.size
        for i in range(1,n):
            pha0 = pha[i-1]
            pha1 = pha[i]
            if pha0 > 90 and pha1 < -90:
                pha[i] += 360
            elif pha0 < -90 and pha1 > 90:
                pha[i] -= 360

        return coh,pha