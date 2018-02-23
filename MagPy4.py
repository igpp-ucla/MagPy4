
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
import matplotlib.ticker as tckr
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
        #self.ui.timeSlider.startValueChanged.connect(self.onStartChanged)
        #self.ui.timeSlider.endValueChanged.connect(self.onEndChanged)
        self.ui.startSlider.valueChanged.connect(self.onStartChanged)
        self.ui.endSlider.valueChanged.connect(self.onEndChanged)
        self.ui.startSlider.sliderReleased.connect(self.setTimes)
        self.ui.endSlider.sliderReleased.connect(self.setTimes)

        self.openFile('C:/Users/jcollins/mmsTestData/L2/merged/2015/09/27/mms15092720')
        self.setupSliders()
        self.calcPlotPoints()
        self.plotFFData()

    def comboChange(self, i):
        print(f'combo changed to: {self.comboBox.itemText(i)}')
            

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

        self.numpoints = self.FID.FFParm["NROWS"].value
        self.tO = self.times[0]
        print(f'tO: {self.tO}')
        last = min(self.numpoints - 1, len(self.times) - 1)
        print(f'last: {last}')
        self.tE = self.times[last]
        print(f'tE: {self.tE}')
        

        #self.plot_.clear()
#       self.setHeaders(self.plot_)
        #self.calibrateMMSPlot(self.plot_, "self.graphs")  # set axis bounds from data
        #self.calibrateMMSPlot(self.plotP, "self.graphs")  # set axis bounds from data
        #self.calibrateMMSPlot(self.plotB, "self.graphs")  # set axis bounds from data
        #self.setTraces()
        #self.updateToolBox()
        #self.magScene.update(self.plotRect)  # refresh plot
        #self._HasData()
        #self.magView.show()
#       self.initSliders()
#       self._HasData()  # enable file dependent widgets
#       print("    END OPEN FILE")
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
        UTCO = FFTIME(self.times[0], Epoch=self.epoch)
        UTCE = FFTIME(self.times[-1], Epoch=self.epoch)
        self.rTime = [UTCO.UTC, UTCE.UTC]
        self.pTime = [UTCO.UTC, UTCE.UTC]
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

        self.ui.startSlider.setMinimum(1)
        self.ui.startSlider.setMaximum(mx)
        self.ui.startSlider.setTickInterval(tick)
        self.ui.startSlider.setSingleStep(tick)
        self.ui.endSlider.setMinimum(1)
        self.ui.endSlider.setMaximum(mx)
        self.ui.endSlider.setTickInterval(tick)
        self.ui.endSlider.setSingleStep(tick)

        #self.ui.timeSlider.setMin(1)
        #self.ui.timeSlider.setMax(mx)
        #self.ui.timeSlider.setStart(1)
        #self.ui.timeSlider.setEnd(mx)

    def onStartChanged(self, val):
        self.iO = val
        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.startSlider.underMouse():
            self.setTimes()

    def onEndChanged(self, val):
        self.iE = val
        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.endSlider.underMouse():
            self.setTimes()

    def setTimes(self):
        axes = self.ui.figure.axes

        self.tO = self.times[self.iO]
        self.tE = self.times[self.iE]

        for ax in axes:
            xmin,xmax,ymin,ymax = ax.axis()
            ax.axis(xmin = self.tO, xmax=self.tE, ymin=ymin, ymax=ymax)
        self.ui.canvas.draw()


    def calcPlotPoints(self):
        #  determine which section to be plotted
        # error cannot use just self.times (data
        times = self.times
        # print("SEARCH TIMES",self.tO, self.tE)
        iO = timeIndex(times, self.tO, dt=self.resolution)
        iE = timeIndex(times, self.tE, dt=self.resolution) + 1
        print(f'iO: {iO}')
        print(f'iE: {iE}')

        # bunch of error debugging
        if iO is None:
            iO = 0 if self.tO < self.times[0] else None
        if iE is None:
            iO = self.times[-1] if self.tE > self.times[-1] else None
        if iO is None and iE is None:
            print("setTraces: OUT OF BOUNDS")
            start = FFTIME(times[0], Epoch=self.epoch)
            stop_ = FFTIME(times[-1], Epoch=self.epoch)
            Start = FFTIME(self.tO, Epoch=self.epoch)
            Stop_ = FFTIME(self.tE, Epoch=self.epoch)
            print("   Valid ", start.UTC, stop_.UTC)
            print("   Selected ", Start.UTC, Stop_.UTC)
            return
        if (iO == iE):
            start = FFTIME(times[0], Epoch=self.epoch)
            stop_ = FFTIME(times[-1], Epoch=self.epoch)
            Start = FFTIME(self.tO, Epoch=self.epoch)
            Stop_ = FFTIME(self.tE, Epoch=self.epoch)
#           print("ARRAY       ", times[0], times[-1])
            print("   Valid    ", start.UTC, stop_.UTC)
            print("   Selected ", Start.UTC, Stop_.UTC)
            print("NO POINTS IN RANGE ", self.tO, self.tE)
            return
        self.xO = iO
        self.xE = iE

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
        xO = self.xO
        xE = self.xE
        X = self.times

        #Xd = FFTIME.list2Date(self.times)

        numCraft = 4
        firstAxis = None
        for j in range(4):   # bx, by, bz, bt (offset into array kinda)
            ax = self.ui.figure.add_subplot(numCraft, 1, j+1, sharex=firstAxis)
            if firstAxis is None:
                firstAxis = ax       

            ax.autoscale(enable=True, axis='both',tight=True)
            ax.set_ymargin(0.05)

            loc = tckr.AutoMinorLocator(5)
            ax.xaxis.set_minor_locator(loc)

            for i in range(numCraft): # for each spacecraft
                col = i * 4 + j
                Y = np.array(self.magData[col])
                if len(Y) <= 1:
                    continue

                color = colors[i]
                ax.plot(X[xO:xE], Y[xO:xE], ',', lw=0.5, snap=True, color = color) #',' makes points
                #ax.plot_date(Xd[xO:xE], Y[xO:xE], ',', lw=0.5, snap=True, color = color) #',' makes points

                #plt.grid(which='minor')
                #plt.tick_params(which='major', length = 7)

                #formatter = tckr.FuncFormatter(self.tickFormat)
                #ax.xaxis.set_major_formatter(formatter)

        #ax = self.ui.figure.add_subplot(numCraft+1, 1, 5)
        #self.timeSlider = Slider(ax, 'time', self.tO, self.tE)
        #self.timeSlider.on_changed(self.updateAxes)

        # refresh canvas
        self.ui.canvas.draw()

    # unused currently. was used by slider code above but then did with pyqt instead
    def updateAxes(self, val):
        axes = self.ui.figure.axes
        #self.tO += (self.tE-self.tO)/2
        pos = self.timeSlider.val
        for ax in axes:
            xmin,xmax,ymin,ymax = ax.axis()
            ax.axis(xmin = pos,xmax=pos+10000, ymin=ymin, ymax=ymax)
        self.ui.canvas.draw()


    # x is value of tick
    # pos is position of tick int time array i think
    def tickFormat(self, x, pos):
         t = FFTIME(x, Epoch=self.epoch)
         return str(t)
        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main = MagPy4Window()
    main.show()

    sys.exit(app.exec_())