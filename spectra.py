

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from FF_Time import FFTIME
from pyqtgraphExtensions import GridGraphicsLayout, LinearGraphicsLayout, LogAxis, BLabelItem
from dataDisplay import UTCQDate
from MagPy4UI import TimeEdit
from spectraUI import SpectraUI, SpectraViewBox
import functools
import time

class Spectra(QtWidgets.QFrame, SpectraUI):
    def __init__(self, window, parent=None):
        super(Spectra, self).__init__(parent)
        self.window = window
        self.ui = SpectraUI()
        self.ui.setupUI(self, window)
        
        self.ui.updateButton.clicked.connect(self.updateSpectra)
        self.ui.bandWidthSpinBox.valueChanged.connect(self.updateSpectra)
        self.ui.separateTracesCheckBox.stateChanged.connect(self.updateSpectra)
        self.ui.aspectLockedCheckBox.stateChanged.connect(self.setAspect)

        self.plotItems = []
        #self.updateSpectra()
        self.window.setLinesVisible(False, 'general')
        self.wasClosed = False

    def updateDelayed(self):
        print('updating')
        QtCore.QTimer.singleShot(500, self.updateSpectra)

    def closeEvent(self, event):
        self.window.endGeneralSelect()
        self.wasClosed = True # setting self.window.spectra=None caused program to crash with no errors.. couldnt figure out why so switched to this

    def setAspect(self):
        for pi in self.plotItems:
            pi.setAspectLocked(self.ui.aspectLockedCheckBox.isChecked())

    # return start and stop indices of selected data
    def getIndices(self, dstr):
        if dstr not in self.indices:
            i0,i1 = self.window.calcDataIndicesFromLines(dstr)
            self.indices[dstr] = (i0,i1)
        return self.indices[dstr]

    def getPoints(self, dstr):
        i0,i1 = self.getIndices(dstr)
        return i1-i0

    def getFreqs(self, dstr):
        N = self.getPoints(dstr)
        if N not in self.freqs:
            self.freqs[N] = self.calculateFreqList(N)
        return self.freqs[N]

    def getfft(self, dstr):
        if dstr not in self.ffts:
            i0,i1 = self.getIndices(dstr)
            data = self.window.getData(dstr)[i0:i1]
            fft = fftpack.rfft(data.tolist())
            self.ffts[dstr] = fft
        return self.ffts[dstr]

    def updateCalculations(self):
        plotInfos = self.window.getSelectedPlotInfo()

        self.indices = {}
        self.freqs = {}
        self.ffts = {}
        self.powers = {}
        self.maxN = 0

        startTime = time.time()

        for li, (strList, penList) in enumerate(plotInfos):
            for i,dstr in enumerate(strList):
                fft = self.getfft(dstr)
                N = self.getPoints(dstr)
                self.maxN = max(self.maxN,N)
                power = self.calculatePower(fft, N)
                self.powers[dstr] = power

        print(f'powers in {time.time() - startTime}')
        startTime = time.time()

        # calculate coherence and phase from pairs
        c0 = self.ui.cohPair0.currentText()
        c1 = self.ui.cohPair1.currentText()
        coh,pha = self.calculateCoherenceAndPhase(self.getfft(c0), self.getfft(c1), self.getPoints(c0))
        self.coh = coh
        self.pha = pha

        print(f'coh/pha in {time.time() - startTime}')

    # some weird stuff is going on in here because there was many conflicts with combining linked y range between plots of each row,
    # log scale, and fixed aspect ratio settings. its all working now pretty good though
    def updateSpectra(self):

        startTime = time.time()
        print('updating spectra')

        self.updateCalculations()

        startTime = time.time()

        plotInfos = self.window.getSelectedPlotInfo()

        self.ui.grid.clear()
        self.ui.labelLayout.clear()
        oneTracePerPlot = self.ui.separateTracesCheckBox.isChecked()
        aspectLocked = self.ui.aspectLockedCheckBox.isChecked()
        numberPlots = 0
        curRow = [] # list of plot items in rows of spectra
        self.plotItems = []
        maxTitleWidth = 0
        for listIndex, (strList,penList) in enumerate(plotInfos):
            for i,dstr in enumerate(strList):
                if i == 0 or oneTracePerPlot:
                    ba = LogAxis(True,True,True,orientation='bottom')
                    la = LogAxis(True,True,True,orientation='left')
                    pi = pg.PlotItem(viewBox = SpectraViewBox(), axisItems={'bottom':ba, 'left':la})
                    if aspectLocked:
                        pi.setAspectLocked()
                    titleString = ''
                    pi.setLogMode(True, True)
                    pi.enableAutoRange(y=False) # disable y auto scaling so doesnt interfere with custom range settings
                    pi.hideButtons() # hide autoscale button
                    numberPlots += 1
                    powers = []

                freq = self.getFreqs(dstr)
                power = self.powers[dstr]
                powers.append(power)

                pen = penList[i]
                titleString = f"{titleString} <span style='color:{pen.color().name()};'>{dstr}</span>"
                pi.plot(freq, power, pen=pen)

                # this part figures out layout of plots into rows depending on settings
                # also links the y scale of each row together
                lastPlotInList = i == len(strList) - 1
                if lastPlotInList or oneTracePerPlot:
                    pi.setLabels(title=titleString, left='Power', bottom='Log Frequency(Hz)')
                    piw = pi.titleLabel._sizeHint[0][0] + 60 # gestimating padding
                    maxTitleWidth = max(maxTitleWidth,piw)
                    self.ui.grid.addItem(pi)
                    self.plotItems.append(pi)
                    curRow.append((pi,powers))
                    if numberPlots % 4 == 0:
                        self.setYRangeForRow(curRow)
                        curRow = []

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

        leftLabel.setHtml('[FILE]<br>[FREQBANDS]<br>[TIME]')
        rightLabel.setHtml(f'{self.window.getFileNameString()}<br>{self.maxN}<br>{startDate} -> {endDate}')
           
        self.ui.labelLayout.addItem(leftLabel)
        self.ui.labelLayout.nextColumn()
        self.ui.labelLayout.addItem(rightLabel)

        print(f'plotted spectra in {time.time() - startTime}')
        startTime = time.time()

        self.updateCohPha()

        print(f'plotted coh/pha in {time.time() - startTime}')
        startTime = time.time()
        ## end of def

    
    def updateCohPha(self):
        c0 = self.ui.cohPair0.currentText()
        c1 = self.ui.cohPair1.currentText()
        freqs = self.getFreqs(c0)

        datas = [[self.ui.cohGrid, self.coh, 'Coherence'],[self.ui.phaGrid, self.pha, 'Phase']]

        for d in datas:
            d[0].clear()
            ba = LogAxis(False,True,False,orientation='bottom')
            la = LogAxis(False,False,False,orientation='left')
            pi = pg.PlotItem(axisItems={'bottom':ba, 'left':la})
            pi.setLogMode(True, False)
            pi.plot(freqs, d[1], pen=self.window.pens[0])
            pi.setLabels(title=f'{d[2]}:  {c0}   vs   {c1}', left=f'{d[2]}', bottom='Log Frequency(Hz)')
            d[0].addItem(pi)


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

    # get pair 1 and 2 from dropdowns as kx and ky
    # should prob put coherence on separate tab with its own ui section
    # also phase too
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
        pha = np.arctan2(qsSum, csSum) * 57.2957 # no idea where this constant came from

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
