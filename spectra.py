

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
from scipy import fftpack
import numpy as np
from LinearGraphicsLayout import LinearGraphicsLayout
from FF_Time import FFTIME

class SpectraUI(object):
    def setupUI(self, Frame):
        Frame.setWindowTitle('Spectra')
        Frame.resize(500,500)

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

        self.oneTracePerCheckBox = QtGui.QCheckBox()
        self.oneTracePerCheckBox.setChecked(True)
        oneTracePerLabel = QtGui.QLabel("One Trace Per Plot")

        self.aspectLockedCheckBox = QtGui.QCheckBox()
        self.aspectLockedCheckBox.setChecked(True)
        aspectLockedLabel = QtGui.QLabel("Lock Aspect Ratio")

        self.updateButton = QtWidgets.QPushButton('Update')
        self.updateButton.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        layout.addLayout(bottomLayout)
        layout.addWidget(self.updateButton)

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.addWidget(bandWidthLabel, 0, 0, 1, 1)
        gridLayout.addWidget(oneTracePerLabel, 1, 0, 1, 1)
        gridLayout.addWidget(aspectLockedLabel, 2, 0, 1, 1)
        gridLayout.addWidget(self.bandWidthSpinBox, 0, 1, 1, 1)
        gridLayout.addWidget(self.oneTracePerCheckBox, 1, 1, 1, 1)
        gridLayout.addWidget(self.aspectLockedCheckBox, 2, 1, 1, 1)
        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        gridLayout.addItem(spacer, 0, 2, 1, 1)

        gridLayout.addWidget(self.updateButton, 0, 3, 1, 1)
        self.updateButton

        layout.addLayout(gridLayout)

#todo show minor ticks on left side
#hide minor tick labels always
class LogAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)

        self.tickFont = QtGui.QFont()
        self.tickFont.setPixelSize(14)
        self.style['maxTextLevel'] = 1 # never have any subtick labels
        self.style['textFillLimits'] = [(0,1.1)] # try to always draw labels
        #self.style['tickLength'] = -10
        #todo: override AxisItem generateDrawSpecs and custom set tick length

    def tickStrings(self, values, scale, spacing):
        return [f'{int(x)}    ' for x in values] # spaces are for eyeballing the auto sizing before rich text override below

    def tickSpacing(self, minVal, maxVal, size):
        #levels = pg.AxisItem.tickSpacing(self,minVal,maxVal,size)
        levels = [(10.0,0),(1.0,0),(0.5,0)]
        return levels

    # overriden from source to be able to have superscript text
    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)
        
        ## draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        p.translate(0.5,0)  ## resolves some damn pixel ambiguity
        
        ## draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)

        ## Draw all text
        if self.tickFont is not None:
            p.setFont(self.tickFont)
        p.setPen(self.pen())
        for rect, flags, text in textSpecs:
            qst = QtGui.QStaticText(f'10<sup>{text}</sup>')
            qst.setTextFormat(QtCore.Qt.RichText)
            p.drawStaticText(rect.left(), rect.top(), qst)
            #p.drawText(rect, flags, text)
            #p.drawRect(rect)

class SpectraInfiniteLine(pg.InfiniteLine):
    def __init__(self, window, index, *args, **kwds):
        pg.InfiniteLine.__init__(self, *args, **kwds)
        self.window = window
        self.index = index

    def mouseDragEvent(self, ev):
        pg.InfiniteLine.mouseDragEvent(self, ev)
        #update all other infinite spectra lines on left or right depending
        if self.movable and ev.button() == QtCore.Qt.LeftButton:
            x = self.getXPos()
            self.window.updateSpectra(self.index,x)

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
        self.ui.oneTracePerCheckBox.stateChanged.connect(self.updateSpectra)
        self.ui.aspectLockedCheckBox.stateChanged.connect(self.updateSpectra)

        self.updateSpectra()

    # todo only send close event if ur current spectra
    def closeEvent(self, event):
        self.window.spectraStep = 0
        self.window.hideAllSpectraLines()

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
        oneTracePerPlot = self.ui.oneTracePerCheckBox.isChecked()
        aspectLocked = self.ui.aspectLockedCheckBox.isChecked()
        numberPlots = 0
        curRow = [] # list of plot items in rows of spectra
        for listIndex, (strList,penList) in enumerate(plotInfos):
            for i,dstr in enumerate(strList):
                if i == 0 or oneTracePerPlot:
                    pi = pg.PlotItem(viewBox = SpectraViewBox(), axisItems={'bottom':LogAxis(orientation='bottom'), 'left':LogAxis(orientation='left')})
                    if aspectLocked:
                        pi.setAspectLocked()
                    titleString = ''
                    pi.setLogMode(True, True)
                    pi.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False) # disabled so doesnt interfere with custom range settings
                    numberPlots += 1
                    powers = []

                # calculate spectra
                data = self.window.DATADICTNOGAPS[dstr][indices[0]:indices[1]] #spectra cant have gaps
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

        labelText = f'FILE {self.window.FID.name.rsplit("/",1)[1]}'

        #get utc start and stop times
        s0 = self.window.spectraRange[0]
        s1 = self.window.spectraRange[1]
        startDate = FFTIME(min(s0,s1), Epoch=self.window.epoch).UTC
        endDate = FFTIME(max(s0,s1), Epoch=self.window.epoch).UTC

        labelText = f'{labelText}<br>FREQ BANDS {self.N}'
        labelText = f'{labelText}<br>TIME {startDate} -> {endDate}'

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


        