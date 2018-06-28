

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

    # todo only send close event if ur current spectra
    def closeEvent(self, event):
        self.window.generalSelectStep = 0
        self.window.setLinesVisible(False, 'general')  # prob bad if open second spectra and start selecting but then close first one

    def setAspect(self):
        for pi in self.plotItems:
            pi.setAspectLocked(self.ui.aspectLockedCheckBox.isChecked())

    # some weird stuff is going on in here because there was many conflicts
    # with combining linked y range between plots of each row, log scale, and
    # fixed aspect ratio settings
    def updateSpectra(self):
        plotInfos = self.window.getSelectedPlotInfo()
        indices = self.window.getTicksFromLines()
        self.N = indices[1] - indices[0]
        #print(self.N)
        freq = self.calculateFreqList()
        if freq is None:
            return

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
                    pi = pg.PlotItem(viewBox = SpectraViewBox(), axisItems={'bottom':LogAxis(orientation='bottom'), 'left':LogAxis(orientation='left')})
                    if aspectLocked:
                        pi.setAspectLocked()
                    titleString = ''
                    pi.setLogMode(True, True)
                    pi.enableAutoRange(y=False) # disable y auto scaling so doesnt interfere with custom range settings
                    pi.hideButtons() # hide autoscale button
                    numberPlots += 1
                    powers = []

                # calculate spectra
                data = self.window.getData(dstr)[indices[0]:indices[1]]
                fft = fftpack.rfft(data.tolist())
                power = self.calculatePower(fft)

                pen = penList[i]
                titleString = f"{titleString} <span style='color:{pen.color().name()};'>{dstr}</span>"
                pi.plot(freq, power, pen=pen)
                powers.append(power)

                # this part figures out layout of plots into rows depending on settings
                # also links the y scale of each row together
                lastPlotInList = i == len(strList) - 1
                if lastPlotInList or oneTracePerPlot:
                    pi.setLabels(title=titleString, left='Power', bottom='Frequency(Hz)')
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
        #get utc start and stop times
        s0 = self.window.times[indices[0]]
        s1 = self.window.times[indices[1]]
        startDate = UTCQDate.removeDOY(FFTIME(min(s0,s1), Epoch=self.window.epoch).UTC)
        endDate = UTCQDate.removeDOY(FFTIME(max(s0,s1), Epoch=self.window.epoch).UTC)

        leftLabel = BLabelItem({'justify':'left'})
        rightLabel = BLabelItem({'justify':'left'})

        leftLabel.setHtml('[FILE]<br>[FREQBANDS]<br>[TIME]')
        rightLabel.setHtml(f'{self.window.FID.name.rsplit("/",1)[1]}<br>{self.N}<br>{startDate} -> {endDate}')
           
        self.ui.labelLayout.addItem(leftLabel)
        self.ui.labelLayout.nextColumn()
        self.ui.labelLayout.addItem(rightLabel)

        ## end of def

    def setYRangeForRow(self, curRow):
        self.ui.grid.nextRow()
        # scale each plot to use same y range
        # the viewRange function was returning incorrect results so had to do manually
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


    def calculateFreqList(self):
        bw = self.ui.bandWidthSpinBox.value()
        km = int((bw + 1.0) * 0.5)
        nband = (self.N - 1) / 2
        half = int(bw / 2)
        nfreq = int(nband - half + 1) #try to match power length
        C = self.N * self.window.resolution
        freq = np.arange(km, nfreq) / C
        #return np.log10(freq)
        if len(freq) < 2:
            print('Proposed spectra plot invalid!\nFrequency list has lass than 2 values')
            return None
        return freq

    def calculatePower(self, fft):
        bw = self.ui.bandWidthSpinBox.value()
        kmo = int((bw + 1) * 0.5)
        nband = (self.N - 1) / 2
        half = int(bw / 2)
        nfreq = int(nband - bw + 1)

        C = 2 * self.window.resolution / self.N
        fsqr = [ft * ft for ft in fft]
        power = [0] * nfreq
        for n in range(nfreq):
            km = kmo + n
            kO = int(km - half)
            kE = int(km + half) + 1

            power[n] = sum(fsqr[kO * 2 - 1:kE * 2 - 1]) / bw * C

        return power

    # get pair 1 and 2 from dropdowns as kx and ky
    # should prob put coherence on separate tab with its own ui section
    # also phase too
    def calculateCoherenceAndPhase(self, fft0, fft1):

        bw = self.ui.bandWidthSpinBox.value()
        kmo = int((bw + 1) * 0.5)
        nband = (self.N - 1) / 2
        half = int(bw / 2)
        nfreq = int(nband - bw + 1)
        kStart = kmo - half
        kSpan = half * 4 + 1
        
        csA = fft0[:-1] * fft1[:-1] + fft0[1:] * fft1[1:]
        qsA = fft0[:-1] * fft1[1:] - fft1[:-1] * fft0[1:]
        pAA = fft0[:-1] * fft0[:-1] + fft0[1:] * fft0[1:]
        pBA = fft1[:-1] * fft1[:-1] + fft1[1:] * fft1[1:]

        csSum = zeros(nFreq)
        qsSum = zeros(nFreq)
        pASum = zeros(nFreq)
        pBSum = zeros(nFreq)

        for n in range(nFreq):
            # km = kmO + n
            # kO = int(km - half)
            # kE = int(km + half) + 1
            KO = (kStart + n) * 2 - 1
            KE = KO + kSpan
            csSum[n] = np.sum(csA[KO:KE:2])
            qsSum[n] = np.sum(qsA[KO:KE:2])
            pASum[n] = np.sum(pAA[KO:KE:2])
            pBSum[n] = np.sum(pBA[KO:KE:2])
        #   for k in range(kO, kE):
        #       i = 2 * k - 1
        #       csSum[n] = csA[i] + csSum[n]
        #       qsSum[n] = qsA[i] + qsSum[n]
        #       pASum[n] = pAA[i] + pASum[n]
        #       pBSum[n] = pBA[i] + pBSum[n]

        coh = (csSum * csSum + qsSum * qsSum) / (pASum * pBSum)
        pha = arctan2(qsSum, csSum) * 57.2957

        return coh,pha


    #def calculateCoherence(self):
    #    # assume that there are no data gaps
    #    # N -> number data points for all
    #    KXKY = {0:(0,1), 1:(0,2), 2:(1,2)}  # coherence pairs
    #    nBAvg = self.ui.bandWidthSpinBox.value()
    #    half = int(nBAvg / 2)
    #    N = self.npts
    #    nband = (N -1) / 2
    #    nFreq = int(nband - nBAvg + 1)
    #    kmO = int((nBAvg + 1) * 0.5)
    #    kStart = kmO - half
    #    kSpan = half * 4 + 1
    #    COH = [0] * 4
    #    PHA = [0] * 4
    #    for panel in range(3):
    #        kx,ky = KXKY[panel]
    #        fft0 = self.fft[kx]
    #        fft1 = self.fft[ky]
    #        csA = fft0[:-1]*fft1[:-1] + fft0[1:]*fft1[1:]
    #        qsA = fft0[:-1]*fft1[1:] - fft1[:-1]*fft0[1:]
    #        pAA = fft0[:-1]*fft0[:-1] + fft0[1:]*fft0[1:]
    #        pBA = fft1[:-1]*fft1[:-1] + fft1[1:]*fft1[1:]
    #        csSum = zeros(nFreq)
    #        qsSum = zeros(nFreq)
    #        pASum = zeros(nFreq)
    #        pBSum = zeros(nFreq)
    #        for n in range(nFreq):
    #            # km = kmO + n
    #            # kO = int(km - half)
    #            # kE = int(km + half) + 1
    #            KO = (kStart + n) * 2 - 1
    #            KE = KO + kSpan
    #            csSum[n] = SUM(csA[KO:KE:2])
    #            qsSum[n] = SUM(qsA[KO:KE:2])
    #            pASum[n] = SUM(pAA[KO:KE:2])
    #            pBSum[n] = SUM(pBA[KO:KE:2])
    #        #   for k in range(kO, kE):
    #        #       i = 2 * k - 1
    #        #       csSum[n] = csA[i] + csSum[n]
    #        #       qsSum[n] = qsA[i] + qsSum[n]
    #        #       pASum[n] = pAA[i] + pASum[n]
    #        #       pBSum[n] = pBA[i] + pBSum[n]
    #        coh = (csSum*csSum + qsSum * qsSum) / (pASum*pBSum)
    #        pha = arctan2(qsSum, csSum) * 57.2957
    #        COH[panel] = coh
    #        # pha0 = pha[:-1]
    #        # pha1 = pha[1:]
    #        # cannot use array operations, history dependencies
    #        # 2015 may 20  corrected when pha < -90
    #        # for i in range(nFreq-1):
    #        #   if (pha0[i] > 90 and pha1[i] < -90):
    #        #       pha[i+1] +=360
    #        #   else: # corrected 2015 may 20
    #        #       if (pha0[i] < -90 and pha1[i] > 90):
    #        #           pha[i+1] -=360
    #        wrapPhase(pha)
    #        #for i in range(1, nFreq):
    #        #   pha0 = pha[i-1]
    #        #   pha1 = pha[i]
    #        #   if (pha0 > 90 and pha1 < -90):
    #        #       pha[i] +=360
    #        #   else:
    #        #       if (pha0 < -90 and pha1 > 90):
    #        #           pha[i] -=360
    #        PHA[panel] = pha
    #    cohAvg = (COH[0]+COH[1]+COH[2])/3.0
    #    phaAvg = (PHA[0]+PHA[1]+PHA[2])/3.0
    #    COH[3] = cohAvg
    #    PHA[3] = phaAvg
    #    self.coher = COH
    #    self.phase = PHA
    #    return
        