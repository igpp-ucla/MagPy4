
import sys
from PyQt5 import QtGui, QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
import matplotlib.ticker as tckr

import random
import numpy as np

from FF_File import timeIndex, FF_STATUS, FF_ID, ColumnStats, arrayToColumns
from FF_Time import FFTIME, leapFile

MAGCOLS = [[1, 2, 3, 4], [9, 10, 11, 12], [17, 18, 19, 20], [25, 26, 27, 28]]
POSCOLS = [5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32]
# BLMCOLS = [33, 34, 35, 36, 37, 38, 39, 40, 41]
# BLMCOLS = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
BLMCOLS = [32, 33, 34, 35, 36, 37, 38]  #  J{X,Y,Z}M J Jpara, Jperp, Angle


class MagPy4Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(MagPy4Window, self).__init__(parent)

        # gives default window options in top right
        self.setWindowFlags(QtCore.Qt.Window)

        # a figure instance to plot on
        #self.figure = plt.figure(figsize=(10,20), dpi=100)
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        self.canvas.mpl_connect('resize_event', self.resizeEvent)

        #self.scroll = QtWidgets.QScrollArea()
        #self.scroll.setWidget(self.canvas)

        #self.scrollBar = NavigationToolbar(self.canvas, )

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        #self.toolbar.actions()[0].triggered.connect(tightness)
        #NavigationToolbar2QT.home = tightness

        # Just some button connected to `plot` method
        self.button = QtWidgets.QPushButton('Plot')
        #self.button.clicked.connect(self.plot)
        self.button.clicked.connect(self.halfAxes)

        self.tightenButton = QtWidgets.QPushButton('Tighten')

        self.tightenButton.clicked.connect(self.tighten)

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        #layout.addWidget(self.scroll)

        horiz = QtWidgets.QHBoxLayout()
        horiz.addWidget(self.button)
        horiz.addWidget(self.tightenButton)
        layout.addLayout(horiz)

        self.setLayout(layout)

        #self.plot() # plot once at beginning

        self.openFile('C:/Users/jcollins/mmsTestData/L2/merged/2015/09/27/mms15092720')
        self.calcPlotPoints()
        self.plotFFData()

    def tighten(self):
        print('hello')
        plt.tight_layout()

    def halfAxes(self):
        axes = self.figure.axes
        self.tO += (self.tE-self.tO)/2
        for ax in axes:
            xmin,xmax,ymin,ymax = ax.axis()
            ax.axis(self.tO,self.tE, ymin, ymax)

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
        print(self.epoch)
        info = self.FID.FFInfo
        err = self.FID.open()
        if err < 0:
            return err, " UNABLE TO OPEN"
        err, mess = self.loadFile()  # read in the file
        if err < 0:
            return err, mess
        self.fTime = [info["FIRST_TIME"].info, info["LAST_TIME"].info]
        self.resolution = self.FID.getResolution()
#       self.numpoints = min(self.numpoints, self.FID.FFParm["NROWS"].value)

        self.numpoints = self.FID.FFParm["NROWS"].value
        self.tO = self.times[0]
        print(self.tO)
        last = min(self.numpoints - 1, len(self.times) - 1)
        self.tE = self.times[last]
        print(self.tE)


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

    def calcPlotPoints(self):
        #  determine which section to be plotted
        # error cannot use just self.times (data
        times = self.times
        # print("SEARCH TIMES",self.tO, self.tE)
        iO = timeIndex(times, self.tO, dt=self.resolution)
        iE = timeIndex(times, self.tE, dt=self.resolution) + 1
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
        self.figure.clear()
        colors = ['r','g','b','black']
        xO = self.xO
        xE = self.xE
        X = self.times

        #Xd = FFTIME.list2Date(self.times)

        numCraft = 4
        firstAxis = None
        for i in range(numCraft): # for each spacecraft
            col = i * 4 + 0 # bx, by, bz, bt  (0 is bx, 1 by, etc)
            Y = np.array(self.magData[col])
            if len(Y) <= 1:
                continue

            ax = self.figure.add_subplot(numCraft, 1, i+1, sharex=firstAxis)
            if firstAxis is None:
                firstAxis = ax       

            ax.autoscale(enable=True, axis='both',tight=True)
            ax.set_ymargin(0.05)

            color = colors[i]
            ax.plot(X[xO:xE], Y[xO:xE], ',', lw=0.5, snap=True, color = color) #',' makes points
            #ax.plot_date(Xd[xO:xE], Y[xO:xE], ',', lw=0.5, snap=True, color = color) #',' makes points

            loc = tckr.AutoMinorLocator(5)
            ax.xaxis.set_minor_locator(loc)
            #plt.grid(which='minor')
            #plt.tick_params(which='major', length = 7)

            #formatter = tckr.FuncFormatter(self.tickFormat)
            #ax.xaxis.set_major_formatter(formatter)


        # refresh canvas
        self.canvas.draw()

    # x is value of tick
    # pos is position of tick int time array i think
    def tickFormat(self, x, pos):
         t = FFTIME(x, Epoch=self.epoch)
         return str(t)
        

    def plot(self):

        # instead of ax.hold(False)
        self.figure.clear()

        rows = 5
        firstAxis = None
        colors = ['r','g','b']
        for i in range(rows):
            # random data
            data = [random.random() for i in range(1000)]

            # create an axis
            ax = self.figure.add_subplot(rows, 1, i+1, sharex=firstAxis)
            if firstAxis is None:
                firstAxis = ax                   

            
            ax.autoscale(enable=True, axis='both',tight=True)
            ax.set_ymargin(0.05)

            # discards the old graph
            # ax.hold(False) # deprecated, see above

            # plot data
            
            color = 'black'
            if i < len(colors):
                color = colors[i]
            #color = colors[len(colors) - 1 if i >= len(colors) else i]

            ax.plot(data, lw=0.5, snap=True, color = color) #',' makes points

        # refresh canvas
        self.canvas.draw()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main = MagPy4Window()
    main.show()

    sys.exit(app.exec_())