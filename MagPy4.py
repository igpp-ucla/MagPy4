
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
import matplotlib.ticker as tckr
import matplotlib.dates as mdates
from matplotlib.widgets import Slider

from MagPy4UI import UI_MagPy4

import random
import numpy as np

from FF_File import timeIndex, FF_STATUS, FF_ID, ColumnStats, arrayToColumns
from FF_Time import FFTIME, leapFile

MAGCOLS = [[1, 2, 3, 4], [9, 10, 11, 12], [17, 18, 19, 20], [25, 26, 27, 28]]
POSCOLS = [5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32]
# BLMCOLS = [33, 34, 35, 36, 37, 38, 39, 40, 41]
# BLMCOLS = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
BLMCOLS = [32, 33, 34, 35, 36, 37, 38]  #  J{X,Y,Z}M J Jpara, Jperp, Angle


class MagPy4Window(QtWidgets.QDialog, UI_MagPy4):
    def __init__(self, parent=None):
        super(MagPy4Window, self).__init__(parent)

        self.ui = UI_MagPy4()
        self.ui.setupUI(self)

        self.ui.button.clicked.connect(self.halfAxes)
        self.ui.tightenButton.clicked.connect(self.tighten)
        self.ui.comboBox.currentIndexChanged.connect(self.comboChange)
        self.ui.drawStyleCombo.currentIndexChanged.connect(self.setLineStyle)
        #self.ui.timeSlider.startValueChanged.connect(self.onStartChanged)
        #self.ui.timeSlider.endValueChanged.connect(self.onEndChanged)
        self.ui.startSlider.valueChanged.connect(self.onStartChanged)
        self.ui.endSlider.valueChanged.connect(self.onEndChanged)
        self.ui.startSlider.sliderReleased.connect(self.setTimes)
        self.ui.endSlider.sliderReleased.connect(self.setTimes)

        self.openFile('C:/Users/jcollins/mmsTestData/L2/merged/2015/09/27/mms15092720')
        self.setupSliders()
        self.markerStyle = ','
        self.lineStyle = ''
        self.plotFFData()

    def comboChange(self, i):
        print(f'combo changed to: {self.comboBox.itemText(i)}')
           

    def setLineStyle(self, i):
        style = self.ui.drawStyleCombo.itemText(i)
        if style == 'dots':
            self.markerStyle = ','
            self.lineStyle = ''
        elif style == 'lines':
            self.markerStyle = ''
            self.lineStyle = '-'
        self.plotFFData()


    def tighten(self):
        print('hello')
        plt.tight_layout()

    def halfAxes(self):
        axes = self.ui.figure.axes
        self.tO += (self.tE-self.tO)/2
        for ax in axes:
            xmin,xmax,ymin,ymax = ax.axis()
            ax.axis(xmin = self.tO,xmax=self.tE, ymin=ymin, ymax=ymax)
        self.ui.canvas.draw()

    def resizeEvent(self, event):
        print('resize detected')
        plt.tight_layout()

    def openFile(self, PATH=None):  # slot when Open pull down is selected
        fFile = PATH
        FID = FF_ID(fFile, status=FF_STATUS.READ | FF_STATUS.EXIST)
        if not FID:
            WARNING(self, "NOT HAPPENING")
            return -1, "BAD"
        #if self.FID is not None:
        #if hasattr(self, FID):
            #self.FID.close()
            #self._NoData()   # purge old panels
        self.FID = FID
        self.epoch = self.FID.getEpoch()
        print(f'epoch: {self.epoch}')
        info = self.FID.FFInfo
        err = self.FID.open()
        if err < 0:
            return err, " UNABLE TO OPEN"
        err, mess = self.loadFile()  # read in the file
        if err < 0:
            return err, mess
        self.fTime = [info["FIRST_TIME"].info, info["LAST_TIME"].info]
        self.resolution = self.FID.getResolution()
        print(f'resolution: {self.resolution}')
#       self.numpoints = min(self.numpoints, self.FID.FFParm["NROWS"].value)

        self.UTCO = FFTIME(self.times[0], Epoch=self.epoch)
        self.UTCE = FFTIME(self.times[-1], Epoch=self.epoch)

        self.numpoints = self.FID.FFParm["NROWS"].value

        self.iO = 0
        self.iE = min(self.numpoints - 1, len(self.times) - 1) #could maybe just be second part of this min not sure
        self.tO = self.times[self.iO]
        self.tE = self.times[self.iE]

        print(f'iO: {self.iO}')
        print(f'iE: {self.iE}')
        print(f'tO: {self.tO}')
        print(f'tE: {self.tE}')
        
        return 1, "FILE " + fFile + "read"

    def loadFile(self):
        if self.FID is None:
            print("PROBLEM", self.ffFile)
            quit()
        nRows = self.FID.getRows()
        records = self.FID.DID.sliceArray(row=1, nRow=nRows)
        self.times = records["time"]
        self.dataByRec = records["data"]
        self.dataByCol = arrayToColumns(records["data"])
        self.data = self.dataByCol    # for FFSpectrar
        self.epoch = self.FID.getEpoch()

        magData = []
#       print(self.dataByRec.shape)
        for i in range(4):
            for j in range(4):
                column = self.dataByRec[:, i * 8 + j]
                magData.append(column)
        BLMData = []
#       print("BLMCOLS", BLMCOLS)
#       print(self.dataByRec.shape)
        for i in BLMCOLS:
            column = self.dataByRec[:, i]
            BLMData.append(column)
        magStats = ColumnStats(magData, self.FID.getError(), NoTime=True)
        BLMStats = ColumnStats(BLMData, self.FID.getError(), NoTime=True)
        self.stats16 = magStats
        self.magData = magData
        self.BLMData = BLMData
        self.stats = self.condenseStats(magStats)   # merge the four spacecraft to one
        self.BLMStats = BLMStats
#       self.fillFileStat()
        #self.updateFileStats()
        return 1, "OKAY"

    # update slider tick amount and timers and labels and stuff based on new file
    def setupSliders(self):
        parm = self.FID.FFParm
        mx = parm["NROWS"].value - 1
        #tick = int(mx / 20) - 1
        tick = 1.0 / self.resolution * 2.0

        self.ui.startSlider.setMinimum(0)
        self.ui.startSlider.setMaximum(mx)
        self.ui.startSlider.setTickInterval(tick)
        self.ui.startSlider.setSingleStep(tick)
        self.ui.endSlider.setMinimum(0)
        self.ui.endSlider.setMaximum(mx)
        self.ui.endSlider.setTickInterval(tick)
        self.ui.endSlider.setSingleStep(tick)
        self.ui.endSlider.setValue(mx)

        utc = self.getUTCString(self.times[self.iO])
        self.ui.startSliderLabel.setText(utc)

        #self.ui.timeSlider.setMin(1)
        #self.ui.timeSlider.setMax(mx)
        #self.ui.timeSlider.setStart(1)
        #self.ui.timeSlider.setEnd(mx)

    def getUTCString(self, time):
        utc = FFTIME(time, Epoch=self.epoch).UTC
        utc = f"{utc.split(' ')[0]} {utc.split(' ',2)[2]}" #cut out day of year?
        return utc

    def onStartChanged(self, val):
        self.iO = val

        utc = self.getUTCString(self.times[self.iO])
        self.ui.startSliderLabel.setText(utc)

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.startSlider.underMouse():
            self.setTimes()

    def onEndChanged(self, val):
        self.iE = val

        utc = self.getUTCString(self.times[self.iE])
        self.ui.endSliderLabel.setText(utc)

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.endSlider.underMouse():
            self.setTimes()

    def setTimes(self):
        axes = self.ui.figure.axes

        self.tO = self.times[self.iO]
        self.tE = self.times[self.iE]

        fixedTics = []
        rng = self.tE - self.tO
        tickCount = 10
        for i in range(tickCount+1):
            n = self.tO + i * (rng/tickCount)
            fixedTics.append(n)

        majorLoc = tckr.FixedLocator(fixedTics)

        for ax in axes:
            xmin,xmax,ymin,ymax = ax.axis()
            ax.axis(xmin = self.tO, xmax=self.tE, ymin=ymin, ymax=ymax)
            ax.xaxis.set_major_locator(majorLoc)

        self.ui.canvas.draw()

    def condenseStats(self, stats, NVectors=4):  # redo stats 4 (bx,by,bz,bt)
        functions = (np.amin, np.amin, np.amax, np.amin, np.amax, np.amin)
        nStats = len(stats)
        l = [0] * NVectors
        for i in range(NVectors):
            l[i] = []
            for j in range(NVectors):
                index = j * 4 + i + 1
                l[i].append(index)
        newStats = np.zeros([nStats, 5])
        for i in range(nStats):
            stat = stats[i]
            newStats[i, 0] = 0
            for j in range(NVectors):
                valuesList = [stat[v] for v in l[j]]
                newValue = functions[i](valuesList)
                newStats[i, j + 1] = newValue
        return newStats

    def plotFFData(self):
        self.ui.figure.clear()
        colors = ['r','g','b','black']

        #Xd = FFTIME.list2Date(self.times)

        numCraft = 4
        loc = tckr.AutoMinorLocator(5)
        formatter = tckr.FuncFormatter(self.tickFormat)
        #majorLoc = tckr.MaxNLocator(nbins=5,integer=False,prune=None)

        fixedTics = []
        rng = self.tE - self.tO
        tickCount = 10
        for i in range(tickCount+1):
            n = self.tO + i * (rng/tickCount)
            fixedTics.append(n)

        majorLoc = tckr.FixedLocator(fixedTics)

        for j in range(4):   # bx, by, bz, bt (offset into array kinda)
            ax = self.ui.figure.add_subplot(numCraft, 1, j+1)

            #ax.autoscale(enable=True, axis='y',tight=None)
            ax.set_ymargin(0.05) # margin inside ax plot

            ax.xaxis.set_minor_locator(loc)
            #formatter = mdates.DateFormatter('%d')
            ax.xaxis.set_major_formatter(formatter)

            ax.xaxis.set_major_locator(majorLoc)
            plt.tick_params(which='major', length = 7)

            for i in range(numCraft): # for each spacecraft
                col = i * 4 + j
                Y = np.array(self.magData[col])
                if len(Y) <= 1:
                    continue
                ax.plot(self.times, Y, marker=self.markerStyle, linestyle=self.lineStyle, lw=0.5, color = colors[i]) #snap=True, 

            # draw horizontal line if crosses zero
            ymin,ymax = plt.ylim()
            if ymin < 0 and ymax > 0:
                ax.axhline(color='r', lw=0.5, dashes=[5,5])

            # move to fit current time
            xmin,xmax,ymin,ymax = ax.axis()
            ax.axis(xmin = self.tO, xmax= self.tE, ymin=ymin, ymax=ymax)

            if j != numCraft-1: # have x axis labels only on bottom
                ax.tick_params(labelbottom='off')  

        #ax = self.ui.figure.add_subplot(numCraft+1, 1, 5)
        #self.timeSlider = Slider(ax, 'time', self.tO, self.tE)
        #self.timeSlider.on_changed(self.updateAxes)

        # refresh canvas
        self.ui.canvas.draw()

    # x is value of tick
    # pos is position of tick int time array i think
    def tickFormat(self, x, pos):
        t = FFTIME(x, Epoch=self.epoch).UTC
        t = str(t)
        t = t.split(' ')[-1]
        t = t.split(':',1)[1]
        diff = self.tE - self.tO
        if diff > 60: # if less than a minute total difference then add milliseconds
            t = t.split('.',1)[0]
        return t
        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main = MagPy4Window()
    main.show()

    sys.exit(app.exec_())