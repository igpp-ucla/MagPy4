
# python 3.6
import os
import sys
# so python looks in paths for these folders too
# maybe make this into actual modules in future
sys.path.insert(0, 'ffPy')
sys.path.insert(0, 'pycdf')

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
import pyqtgraph as pg

from FF_File import timeIndex, FF_STATUS, FF_ID, ColumnStats, arrayToColumns
from FF_Time import FFTIME, leapFile
from MagPy4UI import MagPy4UI
from plotTracer import PlotTracer
from spectra import Spectra, SpectraInfiniteLine
from dataDisplay import DataDisplay, UTCQDate
from edit import Edit
from pyqtgraphExtensions import PlotPointsItem

import time

class MagPy4Window(QtWidgets.QMainWindow, MagPy4UI):
    def __init__(self, app, parent=None):
        super(MagPy4Window, self).__init__(parent)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True) #todo add option to toggle this

        self.app = app
        self.ui = MagPy4UI()
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
        self.ui.actionSpectra.triggered.connect(self.runSpectra)
        self.ui.actionEdit.triggered.connect(self.openEdit)
        self.ui.switchMode.triggered.connect(self.swapMode)
        self.insightMode = False

        # options menu dropdown
        self.ui.scaleYToCurrentTimeAction.triggered.connect(self.updateYRange)
        self.ui.antialiasAction.triggered.connect(self.toggleAntialiasing)
        self.ui.bridgeDataGaps.triggered.connect(self.replotData)
        self.ui.drawPoints.triggered.connect(self.replotData)

        self.lastPlotMatrix = None # used by plot tracer
        self.lastLinkMatrix = None
        self.tracer = None

        self.spectraStep = 0
        self.spectraRange = [0,0]
        self.spectras = []

        self.edit = None

        self.magpyIcon = QtGui.QIcon()
        self.magpyIcon.addFile('images/magPy_blue.ico')
        self.app.setWindowIcon(self.magpyIcon)

        self.marsIcon = QtGui.QIcon()
        self.marsIcon.addFile('images/mars.ico')

        # setup pens
        self.pens = []
        colors = ['#0000ff','#009900','#ff0000','#000000'] # b darkgreen r black
        for c in colors:
            self.pens.append(pg.mkPen(c, width=1))# style=QtCore.Qt.DotLine)
        self.trackerPen = pg.mkPen('#000000', width=1, style=QtCore.Qt.DashLine)

        self.plotItems = []
        self.trackerLines = []
        starterFile = 'testData/mms15092720'
        if os.path.exists(starterFile + '.ffd'):
            self.openFile(starterFile)
            self.plotDataDefault()


    def openTracer(self):
        if self.tracer is not None:
            self.tracer.close()
        self.tracer = PlotTracer(self)

        self.tracer.move(50,400)
        self.tracer.show()

    def runSpectra(self):
        txt = self.ui.actionSpectra.text()
        if txt == 'Spectra':
            for pi in self.plotItems:
                for line in pi.getViewBox().spectLines:
                    line.hide()
            self.spectraStep = 1
            self.ui.actionSpectra.setText('Complete Spectra')
        elif txt == 'Complete Spectra':
            self.ui.actionSpectra.setText('Spectra')
            if self.spectraStep == 1:
                self.spectraStep = 0
                self.hideAllSpectraLines()
            if self.spectraStep == 2:
                spectra = Spectra(self)
                spectra.show()
                self.spectras.append(spectra) # so reference is held until closed

    def openEdit(self):
        if self.edit is not None:
            self.edit.close()
        self.edit = Edit(self)
        self.edit.show()

    def showData(self):
        self.dataDisplay = DataDisplay(self.FID, self.times, self.dataByCol, Title='Flatfile Data')
        self.dataDisplay.show()

    def toggleAntialiasing(self):
        pg.setConfigOption('antialias', self.ui.antialiasAction.isChecked())
        self.replotData()
        for spectra in self.spectras:
            if spectra is not None:
                spectra.updateSpectra()

    # this smooths over data gaps, required for spectra analysis?
    # faster than naive way
    def replaceErrorsWithLast(self,dataOrig):
        data = np.copy(dataOrig)
        segs = self.getSegments(data)
        if len(segs) == 0: # data is pure errors
            return data

        # if first segment doesnt start at 0 
        # set data 0 - X to data at X
        first = segs[0][0]
        if first != 0: 
            data[0:first] = data[first]

        # interate over the gaps in the segment list
        # this could prob be sped up somehow
        for si in range(len(segs) - 1):
            gO = segs[si][1] # start of gap
            gE = segs[si + 1][0] # end of gap
            gSize = gE - gO + 1 # gap size
            for i in range(gO,gE): # calculate gap values by lerping from start to end
                t = (i - gO + 1) / gSize
                data[i] = (1 - t) * data[gO - 1] + t * data[gE]

        # if last segment doesnt end with last index of data
        # then set data X - end based on X
        last = segs[-1][1]
        if last != len(data):
            data[last - 1:len(data)] = data[last - 1]

        return data


    def swapMode(self): #todo: add option to just compile to one version or other with a bool swap as well
        txt = self.ui.switchMode.text()
        self.insightMode = not self.insightMode
        txt = 'Switch to MMS' if self.insightMode else 'Switch to MarsPy'
        self.ui.switchMode.setText(txt)
        self.plotDataDefault()
        self.setWindowTitle('MarsPy' if self.insightMode else 'MagPy4')
        self.app.setWindowIcon(self.marsIcon if self.insightMode else self.magpyIcon)

    def resizeEvent(self, event):
        #print('resize event')
        pass

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
        # errorFlag is usually 1e34 but sometimes less. still huge though
        self.errorFlag = self.FID.FFInfo['ERROR_FLAG'].value
        print(f'error flag: {self.errorFlag}') # not being used currently
        self.errorFlag *= 0.9 # based off FFSpectra.py line 829
        err = self.FID.open()
        if err < 0:
            return err, " UNABLE TO OPEN"
        err, mess = self.loadFile()  # read in the file
        if err < 0:
            return err, mess
        self.resolution = self.FID.getResolution()
        numpoints = self.FID.FFParm["NROWS"].value

        self.iO = 0
        self.iE = min(numpoints - 1, len(self.times) - 1) #could maybe just be second part of this min not sure
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

        if self.edit is not None:
            self.edit.close()
            self.edit = None

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

        self.ORIGDATADICT = {} # maps string to original data array
        self.DATADICT = {}  # stores smoothed version of data and all modifications (dict of dicts)
        self.UNITDICT = {} # maps strings to string unit
        self.IDENTITY = Edit.identity()
        self.MATRIX = Edit.identity() # temp until i add matrix chooser dropdown in plot tracer
        for i, dstr in enumerate(self.DATASTRINGS):
            datas = np.array(self.dataByRec[:,i])
            self.ORIGDATADICT[dstr] = datas
            self.DATADICT[dstr] = { self.IDENTITY : self.replaceErrorsWithLast(datas) }
            self.UNITDICT[dstr] = units[i]

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
            line.setValue(tt + 1)#offset by linewidth so its not visible once released

        dt = UTCQDate.UTC2QDateTime(FFTIME(tt, Epoch=self.epoch).UTC)
        self.ui.endSliderEdit.blockSignals(True) #dont want to trigger callback from this
        self.ui.endSliderEdit.setDateTime(dt)
        self.ui.endSliderEdit.blockSignals(False)

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.endSlider.underMouse():
            self.setTimes()

    # this might be inaccurate...
    def calcTickIndexByTime(self, t):
        perc = (t - self.itO) / (self.itE - self.itO)
        v = int(self.iiE * perc)
        return 0 if v < 0 else self.iiE if v > self.iiE else v

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
        # if giving exact same time index then slightly offset
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

    # setup default 4 axis magdata plot or 3 axis insight plot
    def plotDataDefault(self):
        boolMatrix = []
        keywords = ['BX','BY','BZ']
        links = [[True,True,True]]
        if not self.insightMode:
            keywords.append('BT')
            links[0].append(True)
        
        for kw in keywords:
            boolAxis = []
            for dstr in self.DATASTRINGS:
                if kw.lower() in dstr.lower():
                    boolAxis.append(True)
                else:
                    boolAxis.append(False)
            boolMatrix.append(boolAxis)
                
        self.plotData(boolMatrix, links)

    # todo: have
    def getData(self, dstr):
        return self.DATADICT[dstr][self.MATRIX if self.MATRIX in self.DATADICT[dstr] else self.IDENTITY]

    def getSegments(self, Y):
        segments = np.where(Y >= self.errorFlag)[0].tolist() # find spots where there are errors and make segments
        segments.append(len(Y)) # add one to end so last segment will be added (also if no errors)
        #print(f'SEGMENTS {len(segments)}')
        segList = []
        st = 0 #start index
        for seg in segments: # collect start and end range of each segment
            while st in segments:
                st += 1
            if st >= seg:
                continue
            segList.append((st,seg))
            st = seg + 1
        # returns empty list if data is pure errors
        return segList

    # boolMatrix is same shape as the checkBox matrix but just bools
    def plotData(self, bMatrix=None, fMatrix=None):
        self.ui.glw.clear()

        if bMatrix is None:
            bMatrix = self.lastPlotMatrix
        if fMatrix is None:
            fMatrix = self.lastLinkMatrix

        self.lastPlotMatrix = bMatrix #save bool matrix for latest plot
        self.lastLinkMatrix = fMatrix
        self.plotItems = []
        self.plotDataStrings = [] # for each plot a list of strings for data contained within
        self.plotTracePens = [] # a list of pens for each trace (saved for consistency with spectra)

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
            #vb.enableAutoRange(axis=vb.YAxis)
            #vb.setAutoVisible(y=True)
            #pi.setDownsampling(auto=True)

            # add some lines used to show where time series sliders will zoom
            trackerLine = pg.InfiniteLine(movable=False, angle=90, pos=0, pen=self.trackerPen)
            pi.addItem(trackerLine)
            self.trackerLines.append(trackerLine)

            self.plotItems.append(pi) #save it for ref elsewhere

            pi.hideButtons() # hide autoscale button

            # show top and right axis, but hide labels (they are off by default apparently)
            la = pi.getAxis('left')
            la.style['textFillLimits'] = [(0,1.1)] # no limits basically to force labels by each tick no matter what

            ba = pi.getAxis('bottom')
            ba.style['textFillLimits'] = [(0,1.1)]
            ta = pi.getAxis('top')
            ra = pi.getAxis('right')
            ta.show()
            ra.show()
            ta.setStyle(showValues=False)
            ra.setStyle(showValues=False)

            # only show tick labels on bottom most axis
            if plotIndex != numPlots - 1:
                ba.setStyle(showValues=False)

            # add traces for each data checked for this axis
            axisString = ''
            unit = ''
            plotStrs = []
            tracePens = []
            traceIndex = 0
            traceCount = 0
            for b in bAxis:
                if b:
                    traceCount+=1
            for i,b in enumerate(bAxis):
                if not b: # axis isn't checked so its not gona be on this plot
                    continue

                dstr = self.DATASTRINGS[i]
                u = self.UNITDICT[dstr]

                # figure out which pen to use
                numPens = len(self.pens)
                if traceCount == 1: # if just one trace then base it off which plot
                    penIndex = plotIndex % numPens
                else: # else if base off trace index, capped at pen count
                    penIndex = min(traceIndex,numPens - 1) 
                pen = self.pens[penIndex]

                #save these so spectra can stay synced with main plot
                tracePens.append(pen)
                plotStrs.append(dstr)

                self.plotTrace(pi, dstr, pen)

                # figure out if each axis trace shares same unit
                if unit == '':
                    unit = u
                elif unit != None and unit != u:
                    unit = None

                # build the axis label string for this plot
                axisString += f"<span style='color:{pen.color().name()};'>{dstr}</span>\n" #font-weight: bold

                traceIndex += 1

            self.plotDataStrings.append(plotStrs)
            self.plotTracePens.append(tracePens)

            # draw horizontal line if plot crosses zero
            vr = pi.viewRange()
            if vr[1][0] < 0 and vr[1][1] > 0:
                zeroLine = pg.InfiniteLine(movable=False, angle=0, pos=0)
                zeroLine.setPen(pg.mkPen('#000000', width=1, style=QtCore.Qt.DotLine))
                pi.addItem(zeroLine, ignoreBounds=True)

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
            # could just switch to using <br>'s probably instead
            li.item.setHtml(f"<span style='font-size:{fontSize}pt; white-space:pre;'>{axisString}</span>")
            self.ui.glw.addItem(li)

            self.ui.glw.addItem(pi)
            self.ui.glw.nextRow()

        ## end of main for loop
        self.updateYRange()

    ## end of plot function

    # just redraws the traces and ensures y range is correct
    # dont need to rebuild everything
    # maybe make this part of main plot function?
    def replotData(self):
        for i in range(len(self.plotItems)):
            pi = self.plotItems[i]
            plotStrs = self.plotDataStrings[i]
            pens = self.plotTracePens[i]
            pi.clearPlots()
            for i,dstr in enumerate(plotStrs):
                self.plotTrace(pi, dstr, pens[i])

        self.updateYRange()

    # both plotData and replot use this function internally
    def plotTrace(self, pi, dstr, pen):
        Y = self.getData(dstr)
        if len(Y) <= 1: # not sure if this can happen but just incase
            print(f'Error: insufficient Y data for column "{dstr}"')
            return
        if not self.ui.bridgeDataGaps.isChecked():
            segs = self.getSegments(self.ORIGDATADICT[dstr])                    
            for a,b in segs:
                if self.ui.drawPoints.isChecked():
                    pi.addItem(PlotPointsItem(self.times[a:b], Y[a:b], pen=pen))
                else:
                    pi.plot(self.times[a:b], Y[a:b], pen=pen)
        else:
            if self.ui.drawPoints.isChecked():
                pi.addItem(PlotPointsItem(self.times, Y, pen=pen))
            else:
                pi.plot(self.times, Y, pen = pen)

    # pyqtgraph has y axis linking but not wat is needed
    # this function scales them to have equal sized ranges but not the same actual range
    # also this replicates pyqtgraph setAutoVisible to have scaling for currently selected time vs the whole file
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
                    scaleYToCurrent = self.ui.scaleYToCurrentTimeAction.isChecked()
                    a = self.iO if scaleYToCurrent else 0
                    b = self.iE if scaleYToCurrent else self.iiE
                    if a > b: # so sliders work either way
                        a,b = b,a
                    dstr = self.DATASTRINGS[i]
                    Y = self.getData(dstr)[a:b]
                    minVal = min(minVal, Y.min())
                    maxVal = max(maxVal, Y.max())
            # if range is bad then dont change this plot
            if np.isnan(minVal) or np.isinf(minVal) or np.isnan(maxVal) or np.isinf(maxVal):
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
                    diff = values[i][1] - values[i][0]
                    largest = max(largest,diff)
            # then scale each plot in this row to the range
            for i,checked in enumerate(row):
                if i in skipRangeSet:
                    continue
                if checked:
                    diff = values[i][1] - values[i][0]
                    l2 = (largest - diff) / 2.0
                    self.plotItems[i].setYRange(values[i][0] - l2, values[i][1] + l2)

        # for plot items that aren't apart of a group (and has at least one trace) just scale them to themselves
        # so this effectively acts like they are checked in their own group
        for i,row in enumerate(self.lastPlotMatrix):
            if i not in partOfGroup and i not in skipRangeSet:
                for b in row: # check to see if has at least one trace (not empty row)
                    if b:
                        self.plotItems[i].setYRange(values[i][0], values[i][1])
                        break

    # find points at which each spacecraft crosses this value for the first time after this time
    # [CURRENTLY UNUSED] but the start to the required velocity calculation code
    def findTimes(self, time, value, axis):
        #first need to get correct Y datas based on axis
        dataNames = [f'B{axis}{n}' for n in range(1,5)]
        print(dataNames)

    # updates all spectra lines, index means which set the left or right basically
    # i feel like i did this stupidly but couldnt think of smoother way to do this
    def updateSpectra(self, index, x):
        self.spectraRange[index] = x
        for pi in self.plotItems:
            pi.getViewBox().spectLines[index].setPos(x)

    def hideAllSpectraLines(self, exc=None): # can allow optional viewbox exception
        for pi in self.plotItems:
            vb = pi.getViewBox()
            if exc is not vb:
                for line in pi.getViewBox().spectLines:
                    line.hide()

    # gets indices into time array for selected spectra range
    # makes sure first is less than second
    def getSpectraRangeIndices(self):
        r0 = self.calcTickIndexByTime(self.spectraRange[0])
        r1 = self.calcTickIndexByTime(self.spectraRange[1])
        return [min(r0,r1),max(r0,r1)]

    # based on which plots have active spectra lines, return list for each plot of the datastr and pen for each trace
    def getSpectraPlotInfo(self):
        plotInfo = []
        for i,pi in enumerate(self.plotItems):
            if pi.getViewBox().spectLines[1].isVisible():
                plotInfo.append((self.plotDataStrings[i], self.plotTracePens[i]))
        return plotInfo

    def anySpectraSelected(self):
        for i,pi in enumerate(self.plotItems):
            if pi.getViewBox().spectLines[1].isVisible():
                return True
        return False

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
        self.spectLines = [SpectraInfiniteLine(self.window, 0, movable=True, angle=90, pos=0, pen=pen),
                           SpectraInfiniteLine(self.window, 1, movable=True, angle=90, pos=0, pen=pen)]
        self.dataText = pg.TextItem('test', fill=(255, 255, 255, 200), html='<div style="text-align: center; color:#FFF;">')
        self.dataText.setZValue(10) #draw over the traces
        self.dataText.border = self.window.pens[3] #black pen

        self.addItem(self.vMouseLine, ignoreBounds=True) #so they dont mess up view range
        self.addItem(self.hMouseLine, ignoreBounds=True)
        self.addItem(self.dataText, ignoreBounds=True)

        self.setCrosshairVisible(False)
        for line in self.spectLines:
            self.addItem(line, ignoreBounds = True)
            line.setBounds((self.window.itO, self.window.itE))
            line.hide()
       
    def setCrosshairVisible(self, visible):
        self.vMouseLine.setVisible(visible)
        self.hMouseLine.setVisible(visible)
        self.dataText.setVisible(visible)

    def onLeftClick(self, ev):
         # map the mouse click to data coordinates
        vr = self.viewRange()
        mc = self.mapToView(ev.pos())
        x = mc.x()
        y = mc.y()
        if self.window.spectraStep > 0: # if making a spectra selection
            if self.window.spectraStep == 2:
                if not self.window.anySpectraSelected():
                    self.window.spectraStep = 1

            if self.window.spectraStep == 1: # time range is not yet selected
                if not self.spectLines[0].isVisible():
                    self.spectLines[0].show()
                    self.spectLines[0].setPos(x)
                    self.window.spectraRange[0] = x
                elif not self.spectLines[1].isVisible(): # first pair has been made
                    self.spectLines[1].show()
                    self.spectLines[1].setPos(x)
                    self.window.spectraRange[0] = self.spectLines[0].getPos()[0]
                    self.window.spectraRange[1] = x
                    self.window.spectraStep = 2
                    self.window.hideAllSpectraLines(self)
            elif self.window.spectraStep == 2: # a time range is already selected
                for i,line in enumerate(self.spectLines):
                    line.show()
                    line.setPos(self.window.spectraRange[i])
        else:
            self.setCrosshairVisible(True)

            self.vMouseLine.setPos(x)
            self.hMouseLine.setPos(y)
            self.dataText.setPos(x,y)
            xt = DateAxis.toUTC(x, self.window, True)
            sf = f'{xt}, {y:.4f}'
            print(sf)
            self.dataText.setText(sf)

            #set text anchor based on which quadrant of viewbox dataText crosshair is in
            centerX = vr[0][0] + (vr[0][1] - vr[0][0]) / 2
            centerY = vr[1][0] + (vr[1][1] - vr[1][0]) / 2

            #i think its based on axis ratio of the label so x is about 1/5
            #times of y anchor offset
            self.dataText.setAnchor((-0.04 if x < centerX else 1.04, 1.2 if y < centerY else -0.2))

    def onRightClick(self, ev):
        if self.window.spectraStep > 0: # cancel spectra on this plot
            self.setCrosshairVisible(False)
            for line in self.spectLines:
                line.hide()
        else:
            pg.ViewBox.mouseClickEvent(self,ev) # default right click

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.double(): # double clicking will hide the lines
                self.setCrosshairVisible(False)
                for line in self.spectLines:
                    line.hide()
            else:
               self.onLeftClick(ev)

        else: # asume right click i guess, not sure about middle mouse button click
            self.onRightClick(ev)

        ev.accept()
            
    # mouse drags for now just get turned into clicks, like on release basically, feels nicer
    # technically only need to do this for spectra mode but not being used otherwise so whatever
    def mouseDragEvent(self, ev, axis=None):
        if ev.isFinish(): # on release
            if ev.button() == QtCore.Qt.LeftButton:
                self.onLeftClick(ev)
            elif ev.button() == QtCore.Qt.RightButton:
                self.onRightClick(ev)
        ev.accept()
        #    pg.ViewBox.mouseDragEvent(self, ev)

    def wheelEvent(self, ev, axis=None):
        ev.ignore()

# subclass based off example here:
# https://github.com/ibressler/pyqtgraph/blob/master/examples/customPlot.py
class DateAxis(pg.AxisItem):
    def toUTC(x,window, showMillis=False): # converts seconds since epoch to UTC string
        t = FFTIME(x, Epoch=window.epoch).UTC
        t = str(t)
        t = t.split(' ')[-1]
        t = t.split(':',1)[1]
        if not showMillis and window.tE - window.tO > 10:
            t = t.split('.',1)[0]
        return t

    def tickStrings(self, values, scale, spacing):
        return [DateAxis.toUTC(x,self.window) for x in values]

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    #appName = f'{appName} {version}';
    app.setOrganizationName('IGPP UCLA')
    app.setOrganizationDomain('igpp.ucla.edu')
    app.setApplicationName('MagPy4')
    #app.setApplicationVersion(version)

    main = MagPy4Window(app)
    main.show()

    sys.exit(app.exec_())