
# python 3.6
import os
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np #make sure to use numpy 1.13, later versions have problems with ffPy library
import pyqtgraph as pg

from FF_File import timeIndex, FF_STATUS, FF_ID, ColumnStats, arrayToColumns
from FF_Time import FFTIME, leapFile
from MagPy4UI import UI_MagPy4, UI_AxisTracer


DATATABLE = {
    'BX1': 0, 'BY1': 1, 'BZ1': 2, 'BT1': 3, 'PX1': 4, 'PY1': 5, 'PZ1': 6, 'PT1': 7,
    'BX2': 8, 'BY2': 9, 'BZ2':10, 'BT2':11, 'PX2':12, 'PY2':13, 'PZ2':14, 'PT2':15,
    'BX3':16, 'BY3':17, 'BZ3':18, 'BT3':19, 'PX3':20, 'PY3':21, 'PZ3':22, 'PT3':23,
    'BX4':24, 'BY4':25, 'BZ4':26, 'BT4':27, 'PX4':28, 'PY4':29, 'PZ4':30, 'PT4':31,
    'JXM':32, 'JYM':33, 'JZM':34, 'JTM':35, 'JPARA':36, 'JPERP':37, 'JANGLE':38
    # i think every column is mapped
}
 # dict of lists, key is data string below, value is list of data to plot
DATADICT = {}
# list of each field wanted to plot
DATASTRINGS = ['BX1','BX2','BX3','BX4',
               'BY1','BY2','BY3','BY4',
               'BZ1','BZ2','BZ3','BZ4',
               'BT1','BT2','BT3','BT4',
               'JXM','JYM','JZM','JTM','JPARA','JPERP','JANGLE',
               'VEL']

CALCTABLE = {
    #'VEL':calcVel    
    #'CURL'
    #'PRESSURE'
    #'DENSITY'
}

class MagPy4Window(QtWidgets.QMainWindow, UI_MagPy4):
    def __init__(self, parent=None):
        super(MagPy4Window, self).__init__(parent)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.ui = UI_MagPy4()
        self.ui.setupUI(self)

        self.ui.startSlider.valueChanged.connect(self.onStartChanged)
        self.ui.endSlider.valueChanged.connect(self.onEndChanged)
        self.ui.startSlider.sliderReleased.connect(self.setTimes)
        self.ui.endSlider.sliderReleased.connect(self.setTimes)

        self.ui.actionPlot.triggered.connect(self.openTracer)
        self.ui.actionOpen.triggered.connect(self.openFileDialog)

        self.lastPlotMatrix = None # used by axis tracer
        self.markerStyle = ','
        self.lineStyle = ''

        self.plotItems = []
        starterFile = 'mmsTestData/L2/merged/2015/09/27/mms15092720'
        if os.path.exists(starterFile+'.ffd'):
            self.openFile(starterFile)
            self.plotDataDefault()


    def openTracer(self):
        self.tracer = AxisTracer(self)
        self.tracer.show()

    def resizeEvent(self, event):
        print('resize event')

    def openFileDialog(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, caption="Open Flatfile", options = QtWidgets.QFileDialog.ReadOnly, filter='Flatfiles (*.ffd)')[0]
        if fileName is "":
            print('OPEN FILE FAILED')
            return
        fileName = fileName.rsplit(".", 1)[0]
        if fileName is None:
            print('OPEN FILE FAILED (split)')
            return
        if self.FID is not None:
            self.FID.close()

        self.openFile(fileName) # first elem of tuple is filepath
        self.plotDataDefault()

    def openFile(self, PATH=None):  # slot when Open pull down is selected
        FID = FF_ID(PATH, status=FF_STATUS.READ | FF_STATUS.EXIST)
        if not FID:
            WARNING(self, "NOT HAPPENING")
            return -1, "BAD"

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
#       self.numpoints = min(self.numpoints, self.FID.FFParm["NROWS"].value)

        self.UTCO = FFTIME(self.times[0], Epoch=self.epoch)
        self.UTCE = FFTIME(self.times[-1], Epoch=self.epoch)

        self.numpoints = self.FID.FFParm["NROWS"].value

        self.iO = 0
        self.iE = min(self.numpoints - 1, len(self.times) - 1) #could maybe just be second part of this min not sure
        self.tO = self.times[self.iO]
        self.tE = self.times[self.iE]
        # save initial time range
        self.itO = self.tO
        self.itE = self.tE  

        print(f'resolution: {self.resolution}')
        print(f'iO: {self.iO}')
        print(f'iE: {self.iE}')
        print(f'tO: {self.tO}')
        print(f'tE: {self.tE}')
        
        self.setupSliders()

        return 1, "FILE " + PATH + "read"

    def loadFile(self):
        if self.FID is None:
            print("Error in loadFile (not opened yet I think)")
        nRows = self.FID.getRows()
        records = self.FID.DID.sliceArray(row=1, nRow=nRows)
        self.times = records["time"]
        self.dataByRec = records["data"]
        self.dataByCol = arrayToColumns(records["data"])
        #self.data = self.dataByCol    # for FFSpectrar
        self.epoch = self.FID.getEpoch()

        numRecords = len(self.dataByRec)
        numColumns = len(self.dataByCol)
        print(f'number records: {numRecords}')
        print(f'number columns: {numColumns}')
        for dstr in DATASTRINGS:
            if dstr not in DATATABLE:
                print(f'cant plot {dstr}, not in DATATABLE')
            else:
                i = DATATABLE[dstr]
                DATADICT[dstr] = np.array(self.dataByRec[:,i])

        #print(self.dataByRec.shape)
        #magStats = ColumnStats(magData, self.FID.getError(), NoTime=True)
        #BLMStats = ColumnStats(BLMData, self.FID.getError(), NoTime=True)

        return 1, "OKAY"

    # update slider tick amount and timers and labels and stuff based on new file
    def setupSliders(self):
        parm = self.FID.FFParm
        mx = parm["NROWS"].value - 1
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

        self.tO = self.times[self.iO]
        self.tE = self.times[self.iE]

        for pi in self.plotItems:
            pi.setXRange(self.tO, self.tE, 0.0)

        # can manipulate tick strings here similar to commented code in plotData

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

    # setup default 4 axis magdata plot
    def plotDataDefault(self):
        boolMatrix = []
        for a in range(4):
            boolAxis = []
            for i in range(len(DATASTRINGS)):
                boolAxis.append(False)
            boolMatrix.append(boolAxis)
        for i in range(16):
            boolMatrix[int(i/4)][i] = True
        self.plotData(boolMatrix)

    # boolMatrix is same shape as the checkBox matrix but just bools
    def plotData(self, bMatrix):
        #self.ui.figure.clear()
        self.ui.glw.clear()
        self.lastPlotMatrix = bMatrix #save bool matrix for latest plot
        self.plotItems = []
        #colors = ['r','g','b','black']
        colors = ['#ff0000','#00ff00','#0000ff','#000000']

        # calculate x axis ticks at fixed interval
        #fixedTics = []
        #rng = self.tE - self.tO
        #tickCount = 10
        #for i in range(tickCount+1):
        #    n = self.tO + i * (rng/tickCount)
        #    fixedTics.append(n)
        #majorLoc = tckr.FixedLocator(fixedTics)
        ##majorLoc = tckr.MaxNLocator(nbins=5,integer=False,prune=None)
        ## add minor ticks in between each major
        #loc = tckr.AutoMinorLocator(5)
        #formatter = tckr.FuncFormatter(self.tickFormat)

        numAxes = len(bMatrix)
        plotCount = max(numAxes,4) # always space for at least 4 plots on screen
        for ai,bAxis in enumerate(bMatrix):
            #ax = self.ui.figure.add_subplot(plotCount, 1, ai+1)
            pi = self.ui.glw.addPlot()
            self.ui.glw.nextRow() # each plot on new row
            self.plotItems.append(pi) #save it for use elsewhere
            # show top and right axis, but hide labels (they are off by default apparently)
            la = pi.getAxis('left')
            ba = pi.getAxis('bottom')
            ta = pi.getAxis('top')
            ra = pi.getAxis('right')
            ta.show()
            ra.show()
            ta.setStyle(showValues=False)
            ra.setStyle(showValues=False)

            # only show tick labels on bottom most axis
            if ai != len(bMatrix)-1:
                ba.setStyle(showValues=False)

            # set left axis label up
            la.setStyle(autoExpandTextSpace=False)
            la.label.rotate(90)

            # add traces for each data checked for this axis
            traces = 0 # number of traces on this axis
            axisString = ''
            for i,b in enumerate(bAxis):
                if not b:
                    continue
                c = colors[min(traces,len(colors)-1)] #if more traces on this axis than colors, use last
                dstr = DATASTRINGS[i]
                Y = DATADICT[dstr]
                axisString += f"<span style='color:{c};'>{dstr}</span>\n"
                #axisString += f"<span style='color:{c}; font-weight: bold'>{dstr}</span>"
                #xScale.setLabel(text="<span style='color: #ff0000; font-weight: bold'>X</span> <i>Axis</i>", units="s")
                if len(Y) <= 1:
                    continue
                pi.plot(self.times, Y, pen=c)
                traces += 1

            # draw horizontal line if crosses zero
            #ymin,ymax = ax.get_ylim()
            #if ymin < 0 and ymax > 0:
            #    ax.axhline(color='r', lw=0.5, dashes=[5,5])

            pi.setXRange(self.tO, self.tE, 0.0)
            pi.setLimits(xMin=self.itO, xMax=self.itE)
            #pi.setAspectLocked(True)

            allMagData = True
            for line in axisString.splitlines():
                if not line.startswith('B'):
                    allMagData = False
                    break
            if allMagData and len(axisString) > 0:
                axisString += '[nT]'
            else:
                axisString = axisString[:-1] #remove last newline character

            labelStyle = {'font-weight':'bold', 'white-space':'pre', 'justify':'left'}
            la.setLabel(axisString, units=None, **labelStyle)

        # end of outer for

    # end of def

        

class AxisTracer(QtWidgets.QFrame, UI_AxisTracer):
    def __init__(self, window, parent=None):
        super(AxisTracer, self).__init__(parent)

        self.window = window
        self.ui = UI_AxisTracer()
        self.ui.setupUI(self)
        self.axisCount = 0

        self.ui.drawStyleCombo.currentIndexChanged.connect(self.setLineStyle)
        self.ui.clearButton.clicked.connect(self.clearCheckBoxes)
        self.ui.addAxisButton.clicked.connect(self.addAxis)
        self.ui.removeAxisButton.clicked.connect(self.removeAxis)
        self.ui.plotButton.clicked.connect(self.plotData)

        self.addLabels()

        self.checkBoxes = []
        # lastPlotMatrix is set whenever window does a plot
        if self.window.lastPlotMatrix is not None:
            for i,axis in enumerate(self.window.lastPlotMatrix):
                self.addAxis()
                for j,checked in enumerate(axis):
                    self.checkBoxes[i][j].setChecked(checked)

    
    def setLineStyle(self, i):
        style = self.ui.drawStyleCombo.itemText(i)
        if style == 'dots':
            self.window.markerStyle = ','
            self.window.lineStyle = ''
        elif style == 'lines':
            self.window.markerStyle = ''
            self.window.lineStyle = '-'

    def plotData(self):
        boolMatrix = []
        for cbAxis in self.checkBoxes:
            boolAxis = []
            for cb in cbAxis:
                boolAxis.append(cb.isChecked())
            boolMatrix.append(boolAxis)
        self.window.lastPlotMatrix = boolMatrix
        self.window.plotData(boolMatrix)

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