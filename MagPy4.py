
# python 3.6
import os
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np #make sure to use numpy 1.13, later versions have problems with ffPy library
import pyqtgraph as pg

from FF_File import timeIndex, FF_STATUS, FF_ID, ColumnStats, arrayToColumns
from FF_Time import FFTIME, leapFile
from MagPy4UI import UI_MagPy4
from plotTracer import PlotTracer
from dataDisplay import DataDisplay

from dataLayout import *

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
        self.ui.actionShowData.triggered.connect(self.showData)

        self.lastPlotMatrix = None # used by plot tracer
        self.lastLinkMatrix = None

        # setup pens
        self.pens = []
        colors = ['#0000ff','#00ff00','#ff0000','#000000'] # b g r bl
        for c in colors:
            self.pens.append(pg.mkPen(c, width=1))# style=QtCore.Qt.DotLine)

        self.plotItems = []
        self.trackerLines = []
        starterFile = 'mmsTestData/L2/merged/2015/09/27/mms15092720'
        if os.path.exists(starterFile+'.ffd'):
            self.openFile(starterFile)
            self.plotDataDefault()


    def openTracer(self):
        self.tracer = PlotTracer(self)
        self.tracer.show()

    def showData(self):
        self.dataDisplay = DataDisplay(self.FID, self.times, self.dataByCol, Title='Test title')
        self.dataDisplay.show()

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
        #self.fTime = [info["FIRST_TIME"].info, info["LAST_TIME"].info]
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
        self.ui.startSlider.setValue(0)
        self.ui.endSlider.setMinimum(0)
        self.ui.endSlider.setMaximum(mx)
        self.ui.endSlider.setTickInterval(tick)
        self.ui.endSlider.setSingleStep(tick)
        self.ui.endSlider.setValue(mx)

        utc = self.getUTCString(self.times[self.iO])
        self.ui.startSliderLabel.setText(utc)
        utc = self.getUTCString(self.times[self.iE])
        self.ui.endSliderLabel.setText(utc)

    def getUTCString(self, time):
        utc = FFTIME(time, Epoch=self.epoch).UTC
        utc = f"{utc.split(' ')[0]} {utc.split(' ',2)[2]}" #cut out day of year?
        return utc

    def onStartChanged(self, val):
        self.iO = val

        # move tracker lines to show where new range will be
        time = self.times[self.iO]
        for line in self.trackerLines:
            line.setValue(time)

        utc = self.getUTCString(time)
        self.ui.startSliderLabel.setText(utc)

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.startSlider.underMouse():
            self.setTimes()

    def onEndChanged(self, val):
        self.iE = val

        # move tracker lines to show where new range will be
        time = self.times[self.iE]
        for line in self.trackerLines:
            line.setValue(time+1)#offset by linewidth so its not visible once released

        utc = self.getUTCString(time)

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
        
        self.plotData(boolMatrix, [[True,True,True,True]])

    # boolMatrix is same shape as the checkBox matrix but just bools
    def plotData(self, bMatrix, fMatrix = []):
        #self.ui.figure.clear()
        self.ui.glw.clear()

        self.lastPlotMatrix = bMatrix #save bool matrix for latest plot
        self.lastLinkMatrix = fMatrix
        self.plotItems = []

        # add label for file name at top right
        fileNameLabel = pg.LabelItem()
        fileNameLabel.opts['justify'] = 'right'
        fileNameLabel.item.setHtml(f"<span style='font-size:10pt;'>{self.FID.name}</span>")
        self.ui.glw.nextColumn()
        self.ui.glw.addItem(fileNameLabel)
        self.ui.glw.nextRow()

        self.trackerLines = []

        numAxes = len(bMatrix)
        plotCount = max(numAxes,4) # always space for at least 4 plots on screen
        for pi,bAxis in enumerate(bMatrix):
            axis = DateAxis(orientation='bottom')
            axis.window = self #todo make this class init argument instead probly?
            vb = MagPyViewBox()
            vb.window = self
            pi = pg.PlotItem(viewBox = vb, axisItems={'bottom': axis})

            # add some lines used to show where time series sliders will zoom to
            trackerLine = pg.InfiniteLine(movable=False, angle=90, pos=0)
            trackerLine.setPen(pg.mkPen('#000000', width=1, style=QtCore.Qt.DashLine))
            pi.addItem(trackerLine)
            self.trackerLines.append(trackerLine)

            self.plotItems.append(pi) #save it for ref elsewhere

            pi.hideButtons() # hide autoscale button

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
            if pi != len(bMatrix)-1:
                ba.setStyle(showValues=False)

            # add traces for each data checked for this axis
            traces = 0 # number of traces on this axis
            axisString = ''
            for i,b in enumerate(bAxis):
                if not b:
                    continue
                p = self.pens[min(traces,len(self.pens)-1)] #if more traces on this axis than pens, use last pen
                dstr = DATASTRINGS[i]
                Y = DATADICT[dstr]
                axisString += f"<span style='color:{p.color().name()};'>{dstr}</span>\n" #font-weight: bold
                if len(Y) <= 1:
                    continue
                pi.plot(self.times, Y, pen=p)
                traces += 1
                # trying to figure out how to just plot pixels
                #pi.plot(self.times, Y, pen=None,symbolSize=1, symbolPen=p,symbolBrush=None,pxMode=True)

            # draw horizontal line if plot crosses zero
            vr = pi.viewRange()
            if vr[1][0] < 0 and vr[1][1] > 0:
                zeroLine = pg.InfiniteLine(movable=False, angle=0, pos=0)
                zeroLine.setPen(pg.mkPen('#000000', width=1, style=QtCore.Qt.DotLine))
                pi.addItem(zeroLine)

            # set plot to current range based on time sliders
            pi.setXRange(self.tO, self.tE, 0.0)
            pi.setLimits(xMin=self.itO, xMax=self.itE)

            # add units for mag data
            # should think of more generalized way to do this for sure
            allMagData = True
            for line in axisString.splitlines():
                if not line.split('>')[1].startswith('B'):
                    allMagData = False
                    break
            if allMagData and len(axisString) > 0:
                axisString += "<span style='color:#888888;'>[nT]</span>\n"
            else:
                axisString = axisString[:-1] #remove last newline character

            # add Y axis label based on traces
            li = pg.LabelItem()
            li.item.setHtml(f"<span style='font-size:12pt; white-space:pre;'>{axisString}</span>")
            self.ui.glw.addItem(li)

            self.ui.glw.addItem(pi)
            self.ui.glw.nextRow()


        # based on fixed axis matrix scale groups y axis to eachother
        for row in fMatrix:
            # find largest range
            largest = 0
            for i,b in enumerate(row):
                if b:
                    pi = self.plotItems[i]
                    vr = pi.viewRange()
                    diff = vr[1][1] - vr[1][0] # first is viewrange y max, second is y min
                    largest = max(largest, diff)
            # adjust each plot in this group to largest range
            for i,b in enumerate(row):
                if b:
                    pi = self.plotItems[i]
                    vr = pi.viewRange()
                    diff = vr[1][1] - vr[1][0]
                    l2 = (largest-diff)/2.0
                    pi.setYRange(vr[1][0] - l2, vr[1][1] + l2)

    ##

    #find points at which each spacecraft crosses this value for the first time after this time
    # axis is X Y Z or T
    def findTimes(self, time, value, axis):
        #first need to get correct Y datas based on axis
        dataNames = [f'B{axis}{n}' for n in range(1,5)]
        print(dataNames)

#http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ViewBox/ViewBox.html#ViewBox
class MagPyViewBox(pg.ViewBox): # custom viewbox event handling
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #self.setMouseMode(self.RectMode)

        # add lines to show where left clicks happen
        pen = pg.mkPen('#B600C6', width=1, style=QtCore.Qt.SolidLine)
        self.vMouseLine = pg.InfiniteLine(movable=False, angle=90, pos=0, pen=pen)
        self.hMouseLine = pg.InfiniteLine(movable=False, angle=0, pos=0, pen=pen)

        self.addItem(self.vMouseLine, ignoreBounds=True) #so they dont mess up view range
        self.addItem(self.hMouseLine, ignoreBounds=True)
        self.vMouseLine.hide()
        self.hMouseLine.hide()
        
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.double(): # double clicking will hide the lines
                self.vMouseLine.hide()
                self.hMouseLine.hide()
            else:
                self.vMouseLine.show()
                self.hMouseLine.show()

                # map the mouse click to data coordinates
                vr = self.viewRange()
                mc = self.mapToView(ev.pos())
                x = mc.x()
                y = mc.y()
                print(f'{x} {y}')
                self.vMouseLine.setPos(x)
                self.hMouseLine.setPos(y)

                self.window.findTimes(x,y,'X') #just for testing for now

            ev.accept()
        else:
            pg.ViewBox.mouseClickEvent(self,ev) # default right click

        #if ev.button() == QtCore.Qt.RightButton:
        #    self.autoRange()
            
    def mouseDragEvent(self, ev, axis=None):
        ev.ignore()
        #if ev.button() == QtCore.Qt.RightButton:
        #    ev.ignore()
        #else:
        #    pg.ViewBox.mouseDragEvent(self, ev)

    def wheelEvent(self, ev, axis=None):
        ev.ignore()

# subclass based off example here: https://github.com/ibressler/pyqtgraph/blob/master/examples/customPlot.py
class DateAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        strngs = []
        diff = self.window.tE - self.window.tO
        for x in values:
            t = FFTIME(x, Epoch=self.window.epoch).UTC
            t = str(t)
            t = t.split(' ')[-1]
            t = t.split(':',1)[1]
            if diff > 10: # add milliseconds
                t = t.split('.',1)[0]
            strngs.append(t)

        return strngs

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    try:
        import BUILD_CONSTANTS #set by cxFreeze script
    except ImportError:
        version = "debug"
    else:
        version = str(BUILD_CONSTANTS.version)

    appName = app.applicationName()
    if appName.startswith("python"):
        appName = "MagPy4"
    appName = appName + " " + version;
    app.setOrganizationName("IGPP/UCLA")
    app.setOrganizationDomain("igpp.ucla.edu")
    app.setApplicationName(appName)
    app.setApplicationVersion(version)

    #set app icon
    appIcon = QtGui.QIcon()
    appIcon.addFile("images/magPy_blue.ico")
    app.setWindowIcon(appIcon)

    main = MagPy4Window()
    main.show()

    sys.exit(app.exec_())