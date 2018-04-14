
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
from dataDisplay import DataDisplay, UTCQDate

import time

class MagPy4Window(QtWidgets.QMainWindow, UI_MagPy4):
    def __init__(self, parent=None):
        super(MagPy4Window, self).__init__(parent)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True)

        self.ui = UI_MagPy4()
        self.ui.setupUI(self)

        self.ui.startSlider.valueChanged.connect(self.onStartSliderChanged)
        self.ui.endSlider.valueChanged.connect(self.onEndSliderChanged)
        self.ui.startSlider.sliderReleased.connect(self.setTimes)
        self.ui.endSlider.sliderReleased.connect(self.setTimes)

        self.ui.startSliderEdit.dateTimeChanged.connect(self.onStartEditChanged)
        self.ui.endSliderEdit.dateTimeChanged.connect(self.onEndEditChanged)

        self.ui.actionPlot.triggered.connect(self.openTracer)
        self.ui.actionOpen.triggered.connect(self.openFileDialog)
        self.ui.actionShowData.triggered.connect(self.showData)

        self.ui.scaleYToCurrentTimeCheckBox.stateChanged.connect(self.updateYRange)

        self.lastPlotMatrix = None # used by plot tracer
        self.lastLinkMatrix = None
        self.tracer = None

        # this was testing for segment calculation, this way was easy but slower than current way, saving incase i need later
        ##testa = np.array([5,10,5,0,1,1,.5,28,28,27,1e34,0,0,0,1e34,1e34])
        #testb = [np.array(list(g)) for k,g in groupby(testa, lambda x:x != 1e34) if k]

        # setup pens
        self.pens = []
        colors = ['#0000ff','#00ff00','#ff0000','#000000'] # b g r black
        for c in colors:
            self.pens.append(pg.mkPen(c, width=1))# style=QtCore.Qt.DotLine)

        self.plotItems = []
        self.trackerLines = []
        starterFile = 'mmsTestData/L2/merged/2015/09/27/mms15092720'
        if os.path.exists(starterFile+'.ffd'):
            self.openFile(starterFile)
            self.plotDataDefault()

    def openTracer(self):
        if self.tracer is not None:
            self.tracer.close()
        self.tracer = PlotTracer(self)

        #geo = QtWidgets.QDesktopWidget().availableGeometry()
        #print(geo)

        self.tracer.move(0,500)
        self.tracer.show()

    def showData(self):
        self.dataDisplay = DataDisplay(self.FID, self.times, self.dataByCol, Title='Flatfile Data')
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
        #info = self.FID.FFInfo
        self.errorFlag = self.FID.FFInfo['ERROR_FLAG'].value
        print(f'error flag: {self.errorFlag}')
        err = self.FID.open()
        if err < 0:
            return err, " UNABLE TO OPEN"
        err, mess = self.loadFile()  # read in the file
        if err < 0:
            return err, mess
        self.resolution = self.FID.getResolution()
#       self.numpoints = min(self.numpoints, self.FID.FFParm["NROWS"].value)

        self.numpoints = self.FID.FFParm["NROWS"].value

        self.iO = 0
        self.iE = min(self.numpoints - 1, len(self.times) - 1) #could maybe just be second part of this min not sure
        self.iiE = self.iE
        self.tO = self.times[self.iO]
        self.tE = self.times[self.iE]
        # save initial time range
        self.itO = self.tO
        self.itE = self.tE  

        print(f'resolution: {self.resolution}')
        #print(f'iO: {self.iO}')
        #print(f'iE: {self.iE}')
        print(f'tO: {self.tO}')
        print(f'tE: {self.tE}')
        
        self.setupSliders()

        if self.tracer is not None:
            self.tracer.close()
            self.tracer = None

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

        # ignoring first column because that is time, hence [1:]
        self.DATASTRINGS = self.FID.getColumnDescriptor("NAME")[1:]
        units = self.FID.getColumnDescriptor("UNITS")[1:]

        self.DATADICT = {} # maps strings to tuple of data array and unit
        for i, dstr in enumerate(self.DATASTRINGS):
            self.DATADICT[dstr] = (np.array(self.dataByRec[:,i]),units[i])

        # sorts by units alphabetically
        #datazip = list(zip(self.DATASTRINGS, units))
        #datazip.sort(key=lambda tup: tup[1], reverse=True)
        #self.DATASTRINGS = [tup[0] for tup in datazip]

        return 1, "OKAY"

    # update slider tick amount and timers and labels and stuff based on new file
    def setupSliders(self):
        parm = self.FID.FFParm
        mx = parm["NROWS"].value - 1
        tick = 1.0 / self.resolution * 2.0

        #dont want to trigger callbacks from first plot
        self.ui.startSlider.blockSignals(True)
        self.ui.endSlider.blockSignals(True)
        self.ui.startSliderEdit.blockSignals(True)
        self.ui.endSliderEdit.blockSignals(True)

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

        minDateTime = UTCQDate.UTC2QDateTime(FFTIME(self.times[0], Epoch=self.epoch).UTC)
        maxDateTime = UTCQDate.UTC2QDateTime(FFTIME(self.times[-1], Epoch=self.epoch).UTC)

        self.ui.startSliderEdit.setMinimumDateTime(minDateTime)
        self.ui.startSliderEdit.setMaximumDateTime(maxDateTime)
        self.ui.startSliderEdit.setDateTime(minDateTime)
        self.ui.endSliderEdit.setMinimumDateTime(minDateTime)
        self.ui.endSliderEdit.setMaximumDateTime(maxDateTime)
        self.ui.endSliderEdit.setDateTime(maxDateTime)

        self.ui.startSlider.blockSignals(False)
        self.ui.endSlider.blockSignals(False)
        self.ui.startSliderEdit.blockSignals(False)
        self.ui.endSliderEdit.blockSignals(False)

    def onStartSliderChanged(self, val):
        self.iO = val

        # move tracker lines to show where new range will be
        tt = self.times[self.iO]
        for line in self.trackerLines:
            line.show()
            line.setValue(tt)

        dt = UTCQDate.UTC2QDateTime(FFTIME(tt, Epoch=self.epoch).UTC)
        self.ui.startSliderEdit.blockSignals(True) #dont want to trigger callback from this
        self.ui.startSliderEdit.setDateTime(dt)
        self.ui.startSliderEdit.blockSignals(False)

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.startSlider.underMouse():
            self.setTimes()

    def onEndSliderChanged(self, val):
        self.iE = val

        # move tracker lines to show where new range will be
        tt = self.times[self.iE]
        for line in self.trackerLines:
            line.show()
            line.setValue(tt+1)#offset by linewidth so its not visible once released

        dt = UTCQDate.UTC2QDateTime(FFTIME(tt, Epoch=self.epoch).UTC)
        self.ui.endSliderEdit.blockSignals(True) #dont want to trigger callback from this
        self.ui.endSliderEdit.setDateTime(dt)
        self.ui.endSliderEdit.blockSignals(False)

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.endSlider.underMouse():
            self.setTimes()

    # pretty sure this is accurate
    def calcTickIndexByTime(self, t):
        perc = (t - self.itO) / (self.itE - self.itO)
        return int(self.iiE * perc)

    def setSliderNoCallback(self, slider, i):
        slider.blockSignals(True)
        slider.setValue(i)
        slider.blockSignals(False)

    # this gets called when the start date time edit is changed directly
    def onStartEditChanged(self, val):
        utc = UTCQDate.QDateTime2UTC(val)
        self.iO = self.calcTickIndexByTime(FFTIME(utc, Epoch=self.epoch)._tick)
        self.setSliderNoCallback(self.ui.startSlider, self.iO)
        for line in self.trackerLines:
            line.hide()
        self.setTimes()

    # this gets called when the end date time edit is changed directly
    def onEndEditChanged(self, val):
        utc = UTCQDate.QDateTime2UTC(val)
        self.iE = self.calcTickIndexByTime(FFTIME(utc, Epoch=self.epoch)._tick)
        self.setSliderNoCallback(self.ui.endSlider, self.iE)
        for line in self.trackerLines:
            line.hide()
        self.setTimes()

    def setTimes(self):
        if self.iO == self.iE:
            if self.iE < self.iiE:
                self.iE += 1
            else:
                self.iO -= 1

        self.tO = self.times[self.iO]
        self.tE = self.times[self.iE]
        self.updateXRange()
        self.updateYRange()

    def updateXRange(self):
        for pi in self.plotItems:
            pi.setXRange(self.tO, self.tE, 0.0)

    # setup default 4 axis magdata plot
    def plotDataDefault(self):
        boolMatrix = []
        keyword = ['BX','BY','BZ','BT']
        for a in range(4):
            boolAxis = []
            for dstr in self.DATASTRINGS:
                if keyword[a] in dstr:
                    boolAxis.append(True)
                else:
                    boolAxis.append(False)
            boolMatrix.append(boolAxis)
                
        self.plotData(boolMatrix, [[True,True,True,True]])

    # boolMatrix is same shape as the checkBox matrix but just bools
    def plotData(self, bMatrix, fMatrix = []):
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

        numPlots = len(bMatrix)
        plotCount = max(numPlots,4) # always space for at least 4 plots on screen
        for plotIndex,bAxis in enumerate(bMatrix):
            axis = DateAxis(orientation='bottom')
            axis.window = self
            vb = MagPyViewBox(self)
            pi = pg.PlotItem(viewBox = vb, axisItems={'bottom': axis})
            #pi.setDownsampling(auto=True)

            # add some lines used to show where time series sliders will zoom to
            trackerLine = pg.InfiniteLine(movable=False, angle=90, pos=0)
            trackerLine.setPen(pg.mkPen('#000000', width=1, style=QtCore.Qt.DashLine))
            pi.addItem(trackerLine)
            self.trackerLines.append(trackerLine)

            self.plotItems.append(pi) #save it for ref elsewhere

            pi.hideButtons() # hide autoscale button

            # show top and right axis, but hide labels (they are off by default apparently)
            la = pi.getAxis('left')
            la.style['textFillLimits'] = [(0,2.0)] # no limits basically to force labels by each tick no matter what

            ba = pi.getAxis('bottom')
            ba.style['textFillLimits'] = [(0,2.0)]
            ta = pi.getAxis('top')
            ra = pi.getAxis('right')
            ta.show()
            ra.show()
            ta.setStyle(showValues=False)
            ra.setStyle(showValues=False)

            # only show tick labels on bottom most axis
            if plotIndex != numPlots-1:
                ba.setStyle(showValues=False)

            # add traces for each data checked for this axis
            traces = 0 # number of traces on this axis
            axisString = ''
            unit = ''
            for i,b in enumerate(bAxis):
                if not b: # axis isn't checked so its not gona be on this plot
                    continue

                p = self.pens[min(traces,len(self.pens)-1)] #if more traces on this axis than pens, use last pen
                dstr = self.DATASTRINGS[i]
                dat = self.DATADICT[dstr]
                Y = dat[0] # y data
                u = dat[1] # units

                if len(Y) <= 1: # not sure if this can happen but just incase
                    continue

                # figure out if each axis trace shares same unit
                if unit == '':
                    unit = u
                elif unit != None and unit != u:
                    unit = None

                # build the axis label string for this plot
                axisString += f"<span style='color:{p.color().name()};'>{dstr}</span>\n" #font-weight: bold

                #was using self.errorFlag but sometimes its diffent, prob error in way files were made so just using 1e33
                segments = np.where(Y >= 1e33)[0].tolist() # find spots where there are errors and make segments
                segments.append(len(Y)) # add one to end so last segment will be added (also if no errors)
                #print(f'SEGMENTS {len(segments)}')
                st = 0 #start index
                for seg in segments: # draw each segment of trace
                    while st in segments:
                        st += 1
                    if st >= seg:
                        continue
                    pi.plot(self.times[st:seg], Y[st:seg], pen=p)
                    st = seg+1
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

            # add unit label if each trace on this plot shares same unit
            if unit != None and unit != '':
                axisString += f"<span style='color:#888888;'>[{unit}]</span>\n"
            else:
                axisString = axisString[:-1] #remove last newline character

            fontSize = max(15 - numPlots,7)

            # add Y axis label based on traces
            li = pg.LabelItem()
            li.item.setHtml(f"<span style='font-size:{fontSize}pt; white-space:pre;'>{axisString}</span>")
            self.ui.glw.addItem(li)

            self.ui.glw.addItem(pi)
            self.ui.glw.nextRow()

        self.updateYRange()

        ##
    ##

    # this function is so large because i couldnt figure out how to get pyqtgraph viewbox to return
    # the range of the currently selected region so i had to do myself
    def updateYRange(self):
        if self.lastPlotMatrix is None:
            return
        values = [] # (min,max)
        skipRangeSet = set() # set of plots where the min and max values are infinite so they should be ignored
        # for each plot, find min and max values for current time selection (consider every trace)
        # otherwise just use the whole visible range self.iiO self.iiE
        for plotIndex,bAxis in enumerate(self.lastPlotMatrix):
            minVal = np.inf
            maxVal = -np.inf
            # find min and max values out of all traces on this plot
            for i,checked in enumerate(bAxis):
                if checked:
                    dstr = self.DATASTRINGS[i]
                    dat = self.DATADICT[dstr]
                    scaleYToCurrent = self.ui.scaleYToCurrentTimeCheckBox.isChecked()
                    a = self.iO if scaleYToCurrent else 0
                    b = self.iE if scaleYToCurrent else self.iiE

                    Y = dat[0][a:b] # y data in current range
                    segments = np.where(Y >= 1e33)[0].tolist() # find spots where there are errors and make segments
                    segments.append(len(Y)) # add one to end so last segment will be added (also if no errors)
                    st = 0 #start index
                    for seg in segments:
                        while st in segments:
                            st += 1
                        if st >= seg:
                            continue
                        slice = Y[st:seg]
                        minVal = min(slice.min(), minVal)
                        maxVal = max(slice.max(), maxVal)
                        st = seg+1
            # if range is bad then dont change this
            if np.isnan(minVal) or np.isinf(minVal) or np.isnan(maxVal) or np.isinf(minVal):
                skipRangeSet.add(plotIndex)
            values.append((minVal,maxVal))

        partOfGroup = set()
        for row in self.lastLinkMatrix:
            # for each plot in this link row find largest range
            largest = 0
            for i,checked in enumerate(row):
                if i in skipRangeSet:
                    continue
                if checked:
                    partOfGroup.add(i)
                    diff = values[i][1]-values[i][0]
                    largest = max(largest,diff)
            # then scale each plot in this row to the range
            for i,checked in enumerate(row):
                if i in skipRangeSet:
                    continue
                if checked:
                    diff = values[i][1]-values[i][0]
                    l2 = (largest-diff)/2.0
                    self.plotItems[i].setYRange(values[i][0] - l2, values[i][1] + l2)
        # for plot items that aren't apart of a group (and has at least one trace) just scale them to themselves
        # so this effectively acts like they are checked in their own group
        for i,row in enumerate(self.lastPlotMatrix):
            if i not in partOfGroup and i not in skipRangeSet:
                for b in row: # check to see if has at least one trace (not empty row)
                    if b:
                        self.plotItems[i].setYRange(values[i][0], values[i][1])
                        break

    #find points at which each spacecraft crosses this value for the first time after this time
    # axis is X Y Z or T
    def findTimes(self, time, value, axis):
        #first need to get correct Y datas based on axis
        dataNames = [f'B{axis}{n}' for n in range(1,5)]
        print(dataNames)

# look at the source here to see what functions you might want to override or call
#http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ViewBox/ViewBox.html#ViewBox
class MagPyViewBox(pg.ViewBox): # custom viewbox event handling
    def __init__(self, window, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #self.setMouseMode(self.RectMode)
        self.window = window
        # add lines to show where left clicks happen
        pen = pg.mkPen('#FF0000', width=1, style=QtCore.Qt.DashLine)
        self.vMouseLine = pg.InfiniteLine(movable=False, angle=90, pos=0, pen=pen)
        self.hMouseLine = pg.InfiniteLine(movable=False, angle=0, pos=0, pen=pen)
        self.dataText = pg.TextItem('test', fill=(255, 255, 255, 200), html='<div style="text-align: center; color:#FFF;">')
        self.dataText.setZValue(10) #draw over the traces
        self.dataText.border = self.window.pens[3] #black pen

        self.addItem(self.vMouseLine, ignoreBounds=True) #so they dont mess up view range
        self.addItem(self.hMouseLine, ignoreBounds=True)
        self.addItem(self.dataText, ignoreBounds=True)

        self.vMouseLine.hide()
        self.hMouseLine.hide()
        self.dataText.hide()
        
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.double(): # double clicking will hide the lines
                self.vMouseLine.hide()
                self.hMouseLine.hide()
                self.dataText.hide()
            else:
                self.vMouseLine.show()
                self.hMouseLine.show()
                self.dataText.show()

                # map the mouse click to data coordinates
                vr = self.viewRange()
                mc = self.mapToView(ev.pos())
                x = mc.x()
                y = mc.y()
                self.vMouseLine.setPos(x)
                self.hMouseLine.setPos(y)
                self.dataText.setPos(x,y)
                xt = DateAxis.toUTC(x, self.window, True)
                sf = f'{xt}, {y:.4f}'
                print(sf)
                self.dataText.setText(sf)

                #set text anchor based on which quadrant of viewbox dataText crosshair is in
                centerX = vr[0][0] + (vr[0][1] - vr[0][0])/2
                centerY = vr[1][0] + (vr[1][1] - vr[1][0])/2

                #i think its based on axis ratio of the label so x is about 1/5 times of y anchor offset
                self.dataText.setAnchor((-0.04 if x < centerX else 1.04, 1.2 if y < centerY else -0.2))

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
    def toUTC(x,window, showMillis=False): # converts seconds since epoch to UTC string
        t = FFTIME(x, Epoch=window.epoch).UTC
        t = str(t)
        t = t.split(' ')[-1]
        t = t.split(':',1)[1]
        if not showMillis and window.tE-window.tO > 10:
            t = t.split('.',1)[0]
        return t

    def tickStrings(self, values, scale, spacing):
        return [DateAxis.toUTC(x,self.window) for x in values]

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    #try:
    #    import BUILD_CONSTANTS #was set by old script
    #except ImportError:
    #    version = 'debug'
    #else:
    #    version = str(BUILD_CONSTANTS.version)

    appName = app.applicationName()
    if appName.startswith('python'):
        appName = 'MagPy4'
    #appName = f'{appName} {version}';
    app.setOrganizationName('IGPP UCLA')
    app.setOrganizationDomain('igpp.ucla.edu')
    app.setApplicationName(appName)
    #app.setApplicationVersion(version)

    #set app icon
    appIcon = QtGui.QIcon()
    appIcon.addFile('images/magPy_blue.ico')
    app.setWindowIcon(appIcon)

    main = MagPy4Window()
    main.show()

    sys.exit(app.exec_())