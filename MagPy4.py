
# python 3.6

import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
import matplotlib.ticker as tckr
import matplotlib.dates as mdates
from matplotlib.widgets import Slider

from MagPy4UI import UI_MagPy4, UI_AxisTracer

#import random
#make sure to use numpy 1.13, later versions have problems with ffPy library
import numpy as np 

from FF_File import timeIndex, FF_STATUS, FF_ID, ColumnStats, arrayToColumns
from FF_Time import FFTIME, leapFile

MAGCOLS = [[1, 2, 3, 4], [9, 10, 11, 12], [17, 18, 19, 20], [25, 26, 27, 28]]
POSCOLS = [5, 6, 7, 8, 13, 14, 15, 16, 21, 22, 23, 24, 29, 30, 31, 32]
# BLMCOLS = [33, 34, 35, 36, 37, 38, 39, 40, 41]
# BLMCOLS = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
BLMCOLS = [32, 33, 34, 35, 36, 37, 38]  #  J{X,Y,Z}M J Jpara, Jperp, Angle

DATADICT = {} # dict of lists, key is data string below, value is list of data to plot
DATASTRINGS = ['BX1','BX2','BX3','BX4','BY1','BY2','BY3','BY4','BZ1','BZ2','BZ3','BZ4','BT1','BT2','BT3','BT4','curl','velocity','pressure','density']

class MagPy4Window(QtWidgets.QMainWindow, UI_MagPy4):
    def __init__(self, parent=None):
        super(MagPy4Window, self).__init__(parent)

        self.ui = UI_MagPy4()
        self.ui.setupUI(self)

        self.ui.comboBox.currentIndexChanged.connect(self.comboChange)
        self.ui.drawStyleCombo.currentIndexChanged.connect(self.setLineStyle)
        #self.ui.timeSlider.startValueChanged.connect(self.onStartChanged)
        #self.ui.timeSlider.endValueChanged.connect(self.onEndChanged)
        self.ui.startSlider.valueChanged.connect(self.onStartChanged)
        self.ui.endSlider.valueChanged.connect(self.onEndChanged)
        self.ui.startSlider.sliderReleased.connect(self.setTimes)
        self.ui.endSlider.sliderReleased.connect(self.setTimes)

        #self.ui.actionTest.triggered.connect(self.plotFFDataTest)
        self.ui.actionTest.triggered.connect(self.openTracer)
        self.jTest = 0

        self.openFile('mmsTestData/L2/merged/2015/09/27/mms15092720')
        self.setupSliders()
        self.markerStyle = ','
        self.lineStyle = ''
        #self.plotFFData()

    def comboChange(self, i):
        print(f'combo changed to: {self.ui.comboBox.itemText(i)}')

    def openTracer(self):
        self.tracer = AxisTracer(self)
        self.tracer.show()

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

        magData = [] # can get rid of this eventually with DATADICT holding all fetched data
#       print(self.dataByRec.shape)
        # in order: bx1 by1 bz1 bt1 bx2 by2 etc
        dataAxis = ['X','Y','Z','T']
        for i in range(4):
            for j in range(4):
                column = self.dataByRec[:, i * 8 + j]
                magData.append(column)

                dataStr = f'B{dataAxis[j]}{i+1}'
                DATADICT[dataStr] = np.array(column)

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


    def plotFFDataTest(self):
        #self.ui.figure.clear()
        colors = ['r','g','b','black']
        j = self.jTest
        self.jTest += 1
        if self.jTest > 3:
            self.jTest = 0
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

        axxs = self.ui.figure.axes
        if len(axxs) == 0:
            ax = self.ui.figure.add_subplot(numCraft, 1, j+1)
        else:
            ax = axxs[0]
            ax.clear()

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
        plt.tight_layout()

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

    def plotData(self, cbMatrix):
        self.ui.figure.clear()

        colors = ['r','g','b','black']

        # calculate x axis ticks at fixed interval
        fixedTics = []
        rng = self.tE - self.tO
        tickCount = 10
        for i in range(tickCount+1):
            n = self.tO + i * (rng/tickCount)
            fixedTics.append(n)
        majorLoc = tckr.FixedLocator(fixedTics)
        #majorLoc = tckr.MaxNLocator(nbins=5,integer=False,prune=None)
        # add minor ticks in between each major
        loc = tckr.AutoMinorLocator(5)
        formatter = tckr.FuncFormatter(self.tickFormat)

        numAxes = len(cbMatrix)
        plotCount = max(numAxes,4) # always space for at least 4 plots on screen
        for ai,cbAxis in enumerate(cbMatrix):
            ax = self.ui.figure.add_subplot(plotCount, 1, ai+1)
            
            ax.set_ymargin(0.05) # margin inside ax plot
            ax.xaxis.set_minor_locator(loc)
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(majorLoc)
            plt.tick_params(which='major', length = 7)

            # add traces for each data checked for this axis
            traces = 0 # number of traces on this axis
            for i,cb in enumerate(cbAxis):
                if cb.isChecked():
                    #print(f'{DATASTRINGS[i]}')
                    Y = DATADICT[DATASTRINGS[i]]
                    if len(Y) <= 1:
                        continue
                    ax.plot(self.times, Y, marker=self.markerStyle, linestyle=self.lineStyle, lw=0.5, color = colors[min(traces,len(colors))]) #snap=True, 
                    traces += 1

            # draw horizontal line if crosses zero
            ymin,ymax = ax.get_ylim()
            if ymin < 0 and ymax > 0:
                ax.axhline(color='r', lw=0.5, dashes=[5,5])

            # move to fit current time
            xmin,xmax,ymin,ymax = ax.axis()
            ax.axis(xmin = self.tO, xmax= self.tE, ymin=ymin, ymax=ymax)

            if ai != numAxes-1: # have x axis labels only on bottom (last one to be plotted)
                ax.tick_params(labelbottom='off')  


        # remove extra unused axes
        for i in range(numAxes, len(self.ui.figure.axes)):
            self.ui.figure.axes[i].remove()

        plt.tight_layout()

        # refresh canvas
        self.ui.canvas.draw()

        

class AxisTracer(QtWidgets.QFrame, UI_AxisTracer):
    def __init__(self, window, parent=None):
        super(AxisTracer, self).__init__(parent)

        self.window = window
        self.ui = UI_AxisTracer()
        self.ui.setupUI(self)
        self.axisCount = 0

        self.ui.clearButton.clicked.connect(self.clearCheckBoxes)
        self.ui.addAxisButton.clicked.connect(self.addAxis)
        self.ui.removeAxisButton.clicked.connect(self.removeAxis)
        self.ui.plotButton.clicked.connect(self.plotData)

        self.checkBoxes = []
        self.addLabels()
        for i in range(4):
            self.addAxis()

        # setup default check marks
        for i in range(16):
            self.checkBoxes[int(i/4)][i].setChecked(True)            

    def plotData(self):
        self.window.plotData(self.checkBoxes)

    def clearCheckBoxes(self):
        for row in self.checkBoxes:
            for cb in row:
                cb.setChecked(False)

    def addLabels(self):
        self.labels = []
        for i,dstr in enumerate(DATASTRINGS):
            label = QtWidgets.QLabel()
            label.setText(dstr)
            label.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
            self.ui.grid.addWidget(label,0,i+1,1,1)

    def addAxis(self):
        self.axisCount += 1
        axLabel = QtWidgets.QLabel()
        axLabel.setText(f'Axis{self.axisCount}')
        checkBoxes = []
        a = self.axisCount + 1 # first axis is labels so +1
        self.ui.grid.addWidget(axLabel,a,0,1,1)
        for i,dstr in enumerate(DATASTRINGS):
            checkBox = QtWidgets.QCheckBox()
            checkBoxes.append(checkBox)
            #checkBox.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred))
            self.ui.grid.addWidget(checkBox,a,i+1,1,1)
        self.checkBoxes.append(checkBoxes)

    def removeAxis(self):
        #self.ui.grid.removeRow(self.axisCount)
        if self.axisCount == 0:
            print('no more axes to delete')
            return
        self.axisCount-=1

        count = self.ui.grid.count()
        rowLen = len(DATASTRINGS)+1
        self.checkBoxes = self.checkBoxes[:-1]
        for i in range(count-1,count-rowLen-1,-1): #not sure if i need to iterate backwards even?
            child = self.ui.grid.takeAt(i)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                clearLayout(child.layout())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    main = MagPy4Window()
    main.show()

    sys.exit(app.exec_())