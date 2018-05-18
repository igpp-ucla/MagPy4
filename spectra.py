

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from pyqtgraphExtensions import LinearGraphicsLayout, LogAxis
from FF_Time import FFTIME
from dataDisplay import UTCQDate

class SpectraUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Spectra')
        Frame.resize(750,500)

        layout = QtWidgets.QVBoxLayout(Frame)

        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.gmain = LinearGraphicsLayout() # made this based off pg.GraphicsLayout
        #apparently default is 11, tried getting the margins and they all were zero seems bugged according to pyqtgraph
        self.gmain.setContentsMargins(11,0,11,0) # left top right bottom
        self.gview.setCentralItem(self.gmain)
        self.grid = pg.GraphicsLayout() # based on Qt.GridGraphicsLayout
        self.grid.setContentsMargins(0,0,0,0)
        self.labelLayout = LinearGraphicsLayout(orientation = QtCore.Qt.Horizontal)
        self.labelLayout.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum))
        self.labelLayout.setContentsMargins(0,0,0,0)
        self.gmain.addItem(self.grid)
        self.gmain.addItem(self.labelLayout)
        layout.addWidget(self.gview)

        # bandwidth label and spinbox
        bottomLayout = QtWidgets.QHBoxLayout()
        bandWidthLabel = QtGui.QLabel("Average Bandwidth")
        self.bandWidthSpinBox = QtGui.QSpinBox()
        self.bandWidthSpinBox.setMinimum(1)
        self.bandWidthSpinBox.setSingleStep(2)
        self.bandWidthSpinBox.setProperty("value", 3)

        self.separateTracesCheckBox = QtGui.QCheckBox()
        self.separateTracesCheckBox.setChecked(True)
        separateTraces = QtGui.QLabel("Separate Traces")

        self.aspectLockedCheckBox = QtGui.QCheckBox()
        self.aspectLockedCheckBox.setChecked(True)
        aspectLockedLabel = QtGui.QLabel("Lock Aspect Ratio")

        self.updateButton = QtWidgets.QPushButton('Update')
        self.updateButton.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        layout.addLayout(bottomLayout)
        layout.addWidget(self.updateButton)

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.addWidget(bandWidthLabel, 0, 0, 1, 1)
        gridLayout.addWidget(separateTraces, 1, 0, 1, 1)
        gridLayout.addWidget(aspectLockedLabel, 2, 0, 1, 1)
        gridLayout.addWidget(self.bandWidthSpinBox, 0, 1, 1, 1)
        gridLayout.addWidget(self.separateTracesCheckBox, 1, 1, 1, 1)
        gridLayout.addWidget(self.aspectLockedCheckBox, 2, 1, 1, 1)
        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        gridLayout.addItem(spacer, 0, 2, 1, 1)

        gridLayout.addWidget(self.updateButton, 0, 3, 1, 1)
        self.updateButton

        layout.addLayout(gridLayout)


class SpectraViewBox(pg.ViewBox): # custom viewbox event handling
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)

    # overriding part of this function to get resizing to work correctly with
    # manual y range override and fixed aspect ratio settings
    def updateViewRange(self, forceX=False, forceY=False):
        tr = self.targetRect()
        bounds = self.rect()
        aspect = self.state['aspectLocked']
        if aspect is not False and 0 not in [aspect, tr.height(), bounds.height(), bounds.width()]:
            targetRatio = tr.width()/tr.height() if tr.height() != 0 else 1
            viewRatio = (bounds.width() / bounds.height() if bounds.height() != 0 else 1) / aspect
            viewRatio = 1 if viewRatio == 0 else viewRatio
            if viewRatio > targetRatio:
                pg.ViewBox.updateViewRange(self,False,True) 
                return
        pg.ViewBox.updateViewRange(self,forceX,forceY) #default

class Spectra(QtWidgets.QFrame, SpectraUI):
    def __init__(self, window, parent=None):
        super(Spectra, self).__init__(parent)
        self.window = window
        self.ui = SpectraUI()
        self.ui.setupUI(self)
        
        self.ui.updateButton.clicked.connect(self.updateSpectra)
        self.ui.bandWidthSpinBox.valueChanged.connect(self.updateSpectra)
        self.ui.separateTracesCheckBox.stateChanged.connect(self.updateSpectra)
        self.ui.aspectLockedCheckBox.stateChanged.connect(self.setAspect)

        self.plotItems = []
        self.updateSpectra()

    # todo only send close event if ur current spectra
    def closeEvent(self, event):
        self.window.spectraStep = 0
        self.window.hideAllSpectraLines()

    def setAspect(self):
        for pi in self.plotItems:
            pi.setAspectLocked(self.ui.aspectLockedCheckBox.isChecked())

    # some weird stuff is going on in here because there was many conflicts 
    # with combining linked y range between plots of each row, log scale, and fixed aspect ratio settings
    def updateSpectra(self):
        plotInfos = self.window.getSpectraPlotInfo()
        indices = self.window.getSpectraRangeIndices()
        self.N = indices[1] - indices[0]
        #print(self.N)
        freq = self.calculateFreqList()
        if freq is None:
            return

        self.ui.grid.clear()
        self.ui.grid.currentRow = 0 # clear doesnt get rid of grid layout formatting correctly it seems
        self.ui.grid.currentCol = 0
        self.ui.labelLayout.clear()
        oneTracePerPlot = self.ui.separateTracesCheckBox.isChecked()
        aspectLocked = self.ui.aspectLockedCheckBox.isChecked()
        numberPlots = 0
        curRow = [] # list of plot items in rows of spectra
        self.plotItems = []
        for listIndex, (strList,penList) in enumerate(plotInfos):
            for i,dstr in enumerate(strList):
                if i == 0 or oneTracePerPlot:
                    pi = pg.PlotItem(viewBox = SpectraViewBox(), axisItems={'bottom':LogAxis(orientation='bottom'), 'left':LogAxis(orientation='left')})
                    if aspectLocked:
                        pi.setAspectLocked()
                    titleString = ''
                    pi.setLogMode(True, True)
                    pi.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False) # disabled so doesnt interfere with custom range settings
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
                lastPlotInWhole = listIndex == len(plotInfos)-1
                if lastPlotInList or oneTracePerPlot:
                    pi.setLabels(title=titleString, left='Power', bottom='Frequency(Hz)')
                    self.ui.grid.addItem(pi)
                    self.plotItems.append(pi)
                    curRow.append((pi,powers))
                    if numberPlots % 4 == 0 or (lastPlotInList and lastPlotInWhole):
                        self.ui.grid.nextRow()
                        # scale each plot to use same y range
                        # the viewRange function was returning incorrect results so had to do manually
                        minVal = np.inf
                        maxVal = -np.inf
                        for item in curRow:
                            for pow in item[1]:
                                minVal = min(minVal, min(pow))
                                maxVal = max(maxVal, max(pow))
                        minVal = np.log10(minVal) # since plots are in log mode have to give log version of range
                        maxVal = np.log10(maxVal)
                        for item in curRow:
                            item[0].setYRange(minVal,maxVal)
                        curRow.clear()

        # draw some text info like time range and file and stuff
        li = pg.LabelItem()
        li.opts['justify'] = 'left'

        labelText = f'[FILE] {self.window.FID.name.rsplit("/",1)[1]}'

        #get utc start and stop times
        s0 = self.window.spectraRange[0]
        s1 = self.window.spectraRange[1]
        startDate = UTCQDate.removeDOY(FFTIME(min(s0,s1), Epoch=self.window.epoch).UTC)
        endDate = UTCQDate.removeDOY(FFTIME(max(s0,s1), Epoch=self.window.epoch).UTC)

        labelText = f'{labelText}<br>[FREQ BANDS] {self.N}'
        labelText = f'{labelText}<br>[TIME] {startDate} -> {endDate}'

        li.item.setHtml(labelText)
        self.ui.labelLayout.addItem(li)

        ## end of def

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

    def calculateCoherence(self):
        pass


        