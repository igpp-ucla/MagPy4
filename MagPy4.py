
# python 3.6
import os
import sys
# so python looks in paths for these folders too
# maybe make this into actual modules in future
sys.path.insert(0, 'ffPy')
sys.path.insert(0, 'cdfPy')

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import numpy as np
import pyqtgraph as pg

from FF_File import timeIndex, FF_STATUS, FF_ID, ColumnStats, arrayToColumns
from FF_Time import FFTIME, leapFile
import pycdf

from MagPy4UI import MagPy4UI
from plotMenu import PlotMenu
from spectra import Spectra
from dataDisplay import DataDisplay, UTCQDate
from edit import Edit
from traceStats import TraceStats
from pyqtgraphExtensions import DateAxis, LinkedAxis, PlotPointsItem, PlotDataItemBDS, LinkedInfiniteLine, BLabelItem
from mth import Mth
import bisect
from dataBase import Database

import time
import functools
import multiprocessing as mp
import traceback

class MagPy4Window(QtWidgets.QMainWindow, MagPy4UI):
    def __init__(self, app, parent=None):
        super(MagPy4Window, self).__init__(parent)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('antialias', True) #todo add option to toggle this
        #pg.setConfigOption('useOpenGL', True)

        self.app = app
        self.ui = MagPy4UI()
        self.ui.setupUI(self)

        self.OS = os.name
        if os.name == 'nt':
            self.OS = 'windows'
        print(f'OS: {self.OS}')

        self.ui.startSlider.valueChanged.connect(self.onStartSliderChanged)
        self.ui.endSlider.valueChanged.connect(self.onEndSliderChanged)
        self.ui.startSlider.sliderReleased.connect(self.setTimes)
        self.ui.endSlider.sliderReleased.connect(self.setTimes)

        self.ui.timeEdit.start.dateTimeChanged.connect(self.onStartEditChanged)
        self.ui.timeEdit.end.dateTimeChanged.connect(self.onEndEditChanged)

        self.ui.actionPlot.triggered.connect(self.openPlotMenu)
        self.ui.actionOpenFF.triggered.connect(functools.partial(self.openFileDialog, True))
        self.ui.actionOpenCDF.triggered.connect(functools.partial(self.openFileDialog,False))
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

        self.plotMenu = None
        self.edit = None
        self.dataDisplay = None
        self.spectras = []
        self.traceStats = None

        # these are saves for options for program lifetime
        self.plotMenuCheckBoxMode = False
        self.traceStatsOnTop = False

        # this is where options for plot lifetime are saved
        self.initVariables()

        self.magpyIcon = QtGui.QIcon()
        self.marsIcon = QtGui.QIcon()
        if self.OS == 'mac':
            self.magpyIcon.addFile('images/magPy_blue.hqx')
            self.marsIcon.addFile('images/mars.hqx')
        else:
            self.magpyIcon.addFile('images/magPy_blue.ico')
            self.marsIcon.addFile('images/mars.ico')

        self.app.setWindowIcon(self.magpyIcon)

        # setup pens
        self.pens = []
        colors = ['#0000ff','#009900','#ff0000','#000000'] # b darkgreen r black
        for c in colors:
            self.pens.append(pg.mkPen(c, width=1))# style=QtCore.Qt.DotLine)
        self.trackerPen = pg.mkPen('#000000', width=1, style=QtCore.Qt.DashLine)

        self.plotItems = []
        self.labelItems = []
        self.trackerLines = []
        starterFile = 'testData/mms15092720'
        self.FID = None #thanks yi 5/14/2018
        if os.path.exists(starterFile + '.ffd'):
            self.openFF(starterFile)
            self.plotDataDefault()

    # close any subwindows if main window is closed
    # this should also get called if flatfile changes
    def closeEvent(self, event):
        self.closeAllSubWindows()

    def closeAllSubWindows(self):
        self.closePlotMenu()
        self.closeEdit()
        self.closeData()
        self.closeTraceStats()
        for spectra in self.spectras:
            spectra.close()
        self.spectras = []

    # init variables here that should be reset when file changes
    def initVariables(self):
        self.lastPlotStrings = None
        self.lastPlotLinks = None
        # 0: not selecting, 1 : no lines, 2 : one line, 3+ : two lines
        self.generalSelectStep = 0
        self.generalSelectCanHide = False
        self.editHistory = []
        

    def closePlotMenu(self):
        if self.plotMenu:
            self.plotMenu.close()
            self.plotMenu = None
    def closeEdit(self):
        if self.edit:
            self.edit.close()
            self.edit = None
    def closeData(self):
        if self.dataDisplay:
            self.dataDisplay.close()
            self.dataDisplay = None
    def closeTraceStats(self):
        if self.traceStats:
            self.traceStats.close()
            self.traceStats = None

    def openPlotMenu(self):
        self.closePlotMenu()
        self.plotMenu = PlotMenu(self)

        geo = self.geometry()
        self.plotMenu.move(geo.x() - 8, geo.y() + 100)
        self.plotMenu.show()

    def openEdit(self):
        self.closeTraceStats()
        self.closeEdit()
        self.edit = Edit(self)
        self.edit.show()

    def showData(self):
        self.closeData() # this isnt actually needed it still closes somehow
        self.dataDisplay = DataDisplay(self.FID, self.times, self.dataByCol, self.dataByRec, Title='Flatfile Data')
        self.dataDisplay.show()

    def openTraceStats(self, plotIndex):
        self.closeTraceStats()
        self.traceStats = TraceStats(self, plotIndex)
        self.traceStats.show()

    def runSpectra(self):
        self.closeTraceStats()
        spectra = Spectra(self)
        spectra.show()
        self.spectras.append(spectra) # so reference is held until closed
        self.startGeneralSelect('SPECTRA', '#FF0000', spectra.ui.timeEdit, True)

    def toggleAntialiasing(self):
        pg.setConfigOption('antialias', self.ui.antialiasAction.isChecked())
        self.replotData()
        for spectra in self.spectras:
            if spectra is not None:
                spectra.updateSpectra()

    # this smooths over data gaps, required for spectra analysis?
    # errors before first and after last or just extended from those points
    # errors between are lerped between nearest points
    # todo: try modifying using np.interp (prob faster)
    def interpolateErrors(self,dataOrig):
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
        #print(event.size())
        #self.additionalResizing()
        pass

    def openFileDialog(self, isFlatfile):
        if isFlatfile:
            fileName = QtWidgets.QFileDialog.getOpenFileName(self, caption="Open Flatfile", options = QtWidgets.QFileDialog.ReadOnly, filter='Flatfiles (*.ffd)')[0]
        else:
            fileName = QtWidgets.QFileDialog.getOpenFileName(self, caption="Open Cdf", options = QtWidgets.QFileDialog.ReadOnly, filter='CDF (*.cdf)')[0]

        if fileName is "":
            print('OPEN FILE FAILED')
            return

        if self.FID is not None:
            self.FID.close()

        if isFlatfile:        
            fileName = fileName.rsplit(".", 1)[0]
            if fileName is None:
                print('OPEN FILE FAILED (split)')
                return
            if not self.openFF(fileName):
                return
        else:
            self.openCDF(fileName)

        self.closeAllSubWindows()
        self.initVariables()

        self.plotDataDefault()

    def openFF(self, PATH):  # slot when Open pull down is selected
        FID = FF_ID(PATH, status=FF_STATUS.READ | FF_STATUS.EXIST)
        if not FID:
            print('BAD FLATFILE')
            return False

        self.FID = FID
        self.epoch = self.FID.getEpoch()
        print(f'epoch: {self.epoch}')
        #info = self.FID.FFInfo
        # errorFlag is usually 1e34 but sometimes less. still huge though
        self.errorFlag = self.FID.FFInfo['ERROR_FLAG'].value
        self.errorFlag = 1e31 # overriding for now since the above line is sometimes wrong depending on the file (i think bx saves as 1e31 but doesnt update header)
        print(f'error flag: {self.errorFlag}') # not being used currently
        self.errorFlag *= 0.9 # based off FFSpectra.py line 829
        err = self.FID.open()
        if err < 0:
            print('UNABLE TO OPEN')
            return False
        
        # load flatfile
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

        self.ORIGDATADICT = {} # maps data strings to original data array
        self.DATADICT = {}  # stores smoothed version of data and all edit modifications (dict of dicts)
        self.UNITDICT = {} # maps data strings to unit strings
        self.TIMES = Database() # maps multiple dstrs to appropriate time array
        self.IDENTITY = Mth.identity()
        self.MATRIX = Mth.identity() # maybe add matrix dropdown in plotmenu somewhere
        for i, dstr in enumerate(self.DATASTRINGS):
            datas = np.array(self.dataByRec[:,i])
            self.ORIGDATADICT[dstr] = datas
            self.DATADICT[dstr] = { self.IDENTITY : [self.interpolateErrors(datas),dstr,''] }
            self.UNITDICT[dstr] = units[i]

        # sorts by units alphabetically
        #datazip = list(zip(self.DATASTRINGS, units))
        #datazip.sort(key=lambda tup: tup[1], reverse=True)
        #self.DATASTRINGS = [tup[0] for tup in datazip]

        self.resolution = self.FID.getResolution()
        numpoints = self.FID.FFParm["NROWS"].value

        self.iO = 0
        self.iE = len(self.times) - 1
        self.iiE = self.iE
        self.tO = self.times[self.iO]
        self.tE = self.times[self.iE]
        # save initial time range
        self.itO = self.tO
        self.itE = self.tE  

        #print(f'resolution: {self.resolution}')
        print(f'tO: {self.tO}')
        print(f'tE: {self.tE}')
        print(f'time resolution : {1.0 / self.resolution} Hz')

        # generate resolution data column (to visualize when it isn't constant)
        dstr = 'resolution'
        resData = np.diff(self.times)
        np.append(resData, resData[-1])
        self.DATASTRINGS.append(dstr)
        self.ORIGDATADICT[dstr] = resData
        self.DATADICT[dstr] = { self.IDENTITY : [self.interpolateErrors(resData),dstr,''] }
        self.UNITDICT[dstr] = 'sec'
        resolutions = np.unique(resData)
        print(f'resolutions: {", ".join(map(str,resolutions))}')
        tick = 1.0 / self.resolution * 2.0
        self.ui.setupSliders(tick, numRecords - 1, self.getMinAndMaxDateTime())

        return True

    # currently just reads the columns and data types
    # maybe separate this out into another file
    def openCDF(self,PATH):#,q):
        print(f'opening cdf: {PATH}')
        cdf = pycdf.CDF(PATH)
        if not cdf:
            print('CDF LOAD FAILED')

        datas = []
        epochs = []
        epochLengths = set()
        for key,value in cdf.items():
            print(f'{key} : {value}')
            length = len(value)
            item = (key,length)
            if length <= 1:
                continue

            if 'epoch' in key.lower() or 'time' in key.lower():
                epochs.append(item)
                # not sure how to handle this scenario tho I think it would be rare if anything
                if length in epochLengths:
                    print(f'WARNING: 2 epochs with same length detected')
                epochLengths.add(length)
            else:
                datas.append(item)

        # pair epoch name with lists of data names that use that time
        epochsWithData = {}
        for en, el in epochs:
            edata = []
            for dn, dl in datas:
                if el == dl:
                    edata.append(dn)
            if len(edata) > 0:
                epochsWithData[en] = edata
            else:
                print(f'no data found for this epoch: {en}')

        for key,value in epochsWithData.items():
            print(f'{key} {len(cdf[key])}')
            for v in value:
                print(f'  {v}')


        #attrs = pycdf.zAttrList(cdf[key])
        #if 'FILLVAL' in attrs:
            #print(attrs['FILLVAL'])
        #eArr = MagPy4Window.CDFEpochToTimeTicks(e)
        #esArr = self.CDFEpochToTimeTicks(es)


    def CDFEpochToTimeTicks(cdfEpoch):
        """ convert date data to numpy array of Records"""
        d2tt2 = pycdf.Library().datetime_to_tt2000
        num = len(cdfEpoch)
        arr = np.empty(num)

        #ttmJ2000 = 43167.8160001
        dt = 32.184   # tai - tt time?
        div = 10 ** 9

        rng = range(num)

        arr = [d2tt2(cdfEpoch[i]) / div - dt for i in rng]

        # a lot faster if in another process
        #for i in rng:
            #arr[i] = d2tt2(cdfEpoch[i]) / div - dt
        return arr
    
    def getMinAndMaxDateTime(self):
        minDateTime = UTCQDate.UTC2QDateTime(FFTIME(self.times[0], Epoch=self.epoch).UTC)
        maxDateTime = UTCQDate.UTC2QDateTime(FFTIME(self.times[-1], Epoch=self.epoch).UTC)
        return minDateTime,maxDateTime

    def onStartSliderChanged(self, val):
        self.iO = val

        # move tracker lines to show where new range will be
        tt = self.times[self.iO]
        for line in self.trackerLines:
            line.show()
            line.setValue(tt)

        dt = UTCQDate.UTC2QDateTime(FFTIME(tt, Epoch=self.epoch).UTC)
        self.ui.timeEdit.setStartNoCallback(dt)

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.startSlider.underMouse():
            self.setTimes()

    def onEndSliderChanged(self, val):
        self.iE = val

        # move tracker lines to show where new range will be
        tt = self.times[self.iE]
        for line in self.trackerLines:
            line.show()
            line.setValue(tt + 1) #offset by linewidth so its not visible once released

        dt = UTCQDate.UTC2QDateTime(FFTIME(tt, Epoch=self.epoch).UTC)
        self.ui.timeEdit.setEndNoCallback(dt)

        # send a new time if the user clicks on the bar but not on the sliders
        if not self.ui.startSlider.isSliderDown() and not self.ui.endSlider.isSliderDown() and self.ui.endSlider.underMouse():
            self.setTimes()

    def calcTickIndexByTime(self, t):
        return Mth.clamp(bisect.bisect(self.times,t) - 1, 0, self.iiE)

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

    # in seconds, abs for just incase backwards
    def getSelectedTimeRange(self):
        return abs(self.tE - self.tO)

    def updateXRange(self):
        rng = self.getSelectedTimeRange()
        self.ui.timeLabel.setText('yellow')
        if rng > 30 * 60: # if over half hour show hh:mm:ss
            self.ui.timeLabel.setText('HR:MIN:SEC')
        elif rng > 5: # if over 5 seconds show mm:ss
            self.ui.timeLabel.setText('MIN:SEC')
        else:
            self.ui.timeLabel.setText('MIN:SEC.MS')

        for pi in self.plotItems:
            pi.setXRange(self.tO, self.tE, 0.0)


    # setup default 4 axis magdata plot or 3 axis insight plot
    def getDefaultPlotInfo(self):
        dstrs = []
        keywords = ['BX','BY','BZ']
        links = [[0,1,2]]
        if not self.insightMode:
            keywords.append('BT')
            links[0].append(3)
        
        for kw in keywords:
            row = []
            for dstr in self.DATASTRINGS:
                if kw.lower() in dstr.lower():
                    row.append(dstr)
            dstrs.append(row)

        return dstrs, links

    def plotDataDefault(self):
        dstrs,links = self.getDefaultPlotInfo()
                
        self.plotData(dstrs, links)

    def getDataAndLabel(self, dstr):
        return self.DATADICT[dstr][self.MATRIX if self.MATRIX in self.DATADICT[dstr] else self.IDENTITY]

    def getData(self, dstr):
        return self.getDataAndLabel(dstr)[0]

    # returns the current dstr label (based on edit transformations)
    def getLabel(self, dstr):
        return self.getDataAndLabel(dstr)[1]

    # returns data with error values removed
    def getPrunedData(self, dstr, a, b):
        data = self.getData(dstr)[a:b]
        return data[data < self.errorFlag]

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
    def plotData(self, dataStrings, links):
        self.ui.glw.clear()

        self.lastPlotStrings = dataStrings
        self.lastPlotLinks = links

        self.plotItems = []
        self.labelItems = []

        self.plotTracePens = [] # a list of pens for each trace (saved for consistency with spectra)

        # add label for file name at top right
        fileNameLabel = BLabelItem()
        fileNameLabel.opts['justify'] = 'right'
        fileNameLabel.setHtml(f"<span style='font-size:10pt;'>{self.FID.name}</span>")
        self.ui.glw.nextColumn()
        self.ui.glw.addItem(fileNameLabel)
        self.ui.glw.nextRow()

        self.trackerLines = []

        for plotIndex, dstrs in enumerate(dataStrings):
            axis = DateAxis(orientation='bottom')
            axis.window = self
            vb = MagPyViewBox(self, plotIndex)
            pi = pg.PlotItem(viewBox = vb, axisItems={'bottom': axis, 'left': LinkedAxis(orientation='left') })
            #pi.setClipToView(True) # sometimes cuts off part of plot so kinda trash?
            self.plotItems.append(pi) #save it for ref elsewhere
            vb.enableAutoRange(x=False, y=False) # range is being set manually in both directions
            #vb.setAutoVisible(y=True)
            startDS = 10 if len(self.times) > 10000 else 1 # on start doesnt refresh the downsampling so gotta manual it
            pi.setDownsampling(ds=startDS, auto=True, mode='peak')

            # add some lines used to show where time series sliders will zoom to
            trackerLine = pg.InfiniteLine(movable=False, angle=90, pos=0, pen=self.trackerPen)
            pi.addItem(trackerLine)
            self.trackerLines.append(trackerLine)

            # add horizontal zero line
            zeroLine = pg.InfiniteLine(movable=False, angle=0, pos=0)
            zeroLine.setPen(pg.mkPen('#000000', width=1, style=QtCore.Qt.DotLine))
            pi.addItem(zeroLine, ignoreBounds=True)

            pi.hideButtons() # hide autoscale button

            # show top and right axis, but hide labels (they are off by default apparently)
            la = pi.getAxis('left')
            la.style['textFillLimits'] = [(0,1.1)] # no limits basically to force labels by each tick no matter what
            #la.setWidth(50) # this also kinda works but a little space wasteful, saving as reminder incase dynamic solution messes up

            ba = pi.getAxis('bottom')
            #ba.style['textFillLimits'] = [(0,1.1)]
            ta = pi.getAxis('top')
            ra = pi.getAxis('right')
            ta.show()
            ra.show()
            ta.setStyle(showValues=False)
            ra.setStyle(showValues=False)

            # only show tick labels on bottom most axis
            if plotIndex != len(dataStrings) - 1:
                ba.setStyle(showValues=False)

            tracePens = []
            # add traces on this plot for each dstr
            for i,dstr in enumerate(dstrs):
                u = self.UNITDICT[dstr]

                # figure out which pen to use
                numPens = len(self.pens)
                if len(dstrs) == 1: # if just one trace then base it off which plot
                    penIndex = plotIndex % numPens
                else: # else if base off trace index, capped at pen count
                    penIndex = min(i,numPens - 1) 
                pen = self.pens[penIndex]

                #save pens so spectra can stay synced with main plot
                tracePens.append(pen)

                self.plotTrace(pi, dstr, pen)

            self.plotTracePens.append(tracePens)

            # set plot to current range based on time sliders
            pi.setXRange(self.tO, self.tE, 0.0)
            pi.setLimits(xMin=self.itO, xMax=self.itE)

            # add Y axis label based on traces (label gets set below)
            li = BLabelItem()
            self.labelItems.append(li)

            self.ui.glw.addItem(li)
            self.ui.glw.addItem(pi)
            self.ui.glw.nextRow()

        self.additionalResizing()

        ## end of main for loop
        self.updateXRange()
        self.updateYRange()
            
    ## end of plot function

    def setYAxisLabels(self):
        plots = len(self.plotItems)
        for dstrs,pens,li in zip(self.lastPlotStrings,self.plotTracePens,self.labelItems):
            traceCount = len(dstrs)
            alab = ''
            unit = ''
            for dstr,pen in zip(dstrs,pens):
                u = self.UNITDICT[dstr]
                # figure out if each axis trace shares same unit
                if unit == '':
                    unit = u
                elif unit != None and unit != u:
                    unit = None

                l = self.getLabel(dstr)
                alab += f"<span style='color:{pen.color().name()};'>{l}</span>\n"

            # add unit label if each trace on this plot shares same unit
            if unit != None and unit != '':
                alab += f"<span style='color:#888888;'>[{unit}]</span>\n"
            else:
                alab = alab[:-1] #remove last newline character

            fontSize = self.suggestedFontSize
            if traceCount > plots and plots > 1:
                fontSize -= (traceCount - plots) * (1.0 / min(4, plots) + 0.35)
            fontSize = min(16, max(fontSize,4))
            li.setHtml(f"<span style='font-size:{fontSize}pt; white-space:pre;'>{alab}</span>")

    # trying to correctly estimate size of rows. last one needs to be a bit larger since bottom axis
    # this is super hacked but it seems to work okay
    def additionalResizing(self):
        # may want to redo with viewGeometry of plots in mind, might be more consistent than fontsize stuff on mac for example
        #for pi in self.plotItems:
        #    print(pi.viewGeometry())

        #print('additionally resizing')

        qggl = self.ui.glw.layout
        rows = qggl.rowCount()
        plots = rows - 1

        width = self.ui.glw.viewRect().width()
        height = self.ui.glw.viewRect().height()

        self.ui.timeLabel.setPos(width / 2, height - 30)

        # set font size, based on viewsize and plot number
        # very hardcoded just based on nice numbers i found. the goal here is for the labels to be smaller so the plotItems will dictate scaling
        # if these labelitems are larger they will cause qgridlayout to stretch and the plots wont be same height which is bad
        if plots > 0:
            self.suggestedFontSize = height / plots * 0.065
            self.setYAxisLabels()

        if plots <= 1: # if just one plot dont need handle multiplot problems
            return

        # all of this depends on screen resolution so this is a pretty bad fix for now
        bPadding = 20 # for bottom axis text
        #plotSpacing = 10 if self.OS == 'mac' else 7
        plotSpacing = 0.01 * height   #7/689 is about 1% of screen

        edgePadding = 60
        height -= bPadding + edgePadding + (plots - 1) * plotSpacing # the spaces in between plots hence the -1

        # when running out of room bottom plot starts shrinking
        # pyqtgraph is doing some additional resizing independent of the underlying qt layout, but cant figure out where (this kind of works)
        for i in range(rows):
            if i == 0:
                continue

            h = height / plots if i < rows - 1 else height / plots + bPadding
            qggl.setRowFixedHeight(i, h)
            qggl.setRowMinimumHeight(i, h)
            qggl.setRowMaximumHeight(i, h) 

        # this ensures the plots always have same physical width (important because they all share bottom time axis)
        # it works but at start of plot its wrong because the bounds and stuff isnt correct yet
        # this is a recurring problem in multiple areas but not sure how to figure out yet and its not that bad in this case
        maxWidth = 0
        for pi in self.plotItems:
            la = pi.getAxis('left')
            maxWidth = max(maxWidth, la.calcDesiredWidth())
        for pi in self.plotItems:
            la = pi.getAxis('left')
            la.setWidth(maxWidth)
        #print(maxWidth)
        


    # just redraws the traces and ensures y range is correct
    # dont need to rebuild everything
    # maybe make this part of main plot function?
    def replotData(self):
        for i in range(len(self.plotItems)):
            pi = self.plotItems[i]
            plotStrs = self.lastPlotStrings[i]
            pens = self.plotTracePens[i]
            pi.clearPlots()
            for i,dstr in enumerate(plotStrs):
                self.plotTrace(pi, dstr, pens[i])
        self.setYAxisLabels()
        self.updateYRange()
        #self.ui.glw.layoutChanged()

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
                    #pi.plot(self.times[a:b], Y[a:b], pen=pen)
                    pi.addItem(PlotDataItemBDS(self.times[a:b], Y[a:b], pen=pen))
        else:
            if self.ui.drawPoints.isChecked():
                pi.addItem(PlotPointsItem(self.times, Y, pen=pen))
            else:
                #pi.plot(self.times, Y, pen = pen)
                pi.addItem(PlotDataItemBDS(self.times, Y, pen = pen))

    # pyqtgraph has y axis linking but not wat is needed
    # this function scales them to have equal sized ranges but not the same actual range
    # also this replicates pyqtgraph setAutoVisible to have scaling for currently selected time vs the whole file
    def updateYRange(self):
        if self.lastPlotStrings is None or len(self.lastPlotStrings) == 0:
            return
        values = [] # (min,max)
        skipRangeSet = set() # set of plots where the min and max values are infinite so they should be ignored
        # for each plot, find min and max values for current time selection (consider every trace)
        # otherwise just use the whole visible range self.iiO self.iiE
        for plotIndex, dstrs in enumerate(self.lastPlotStrings):
            minVal = np.inf
            maxVal = -np.inf
            # find min and max values out of all traces on this plot
            for dstr in dstrs:
                scaleYToCurrent = self.ui.scaleYToCurrentTimeAction.isChecked()
                a = self.iO if scaleYToCurrent else 0
                b = self.iE if scaleYToCurrent else self.iiE
                if a > b: # so sliders work either way
                    a,b = b,a
                Y = self.getData(dstr)[a:b]
                minVal = min(minVal, Y.min())
                maxVal = max(maxVal, Y.max())
            # if range is bad then dont change this plot
            if np.isnan(minVal) or np.isinf(minVal) or np.isnan(maxVal) or np.isinf(maxVal):
                skipRangeSet.add(plotIndex)
            values.append((minVal,maxVal))

        for row in self.lastPlotLinks:
            # find largest range in group
            largest = 0
            for i in row:
                if i in skipRangeSet:
                    continue
                diff = values[i][1] - values[i][0]
                largest = max(largest,diff)

            # then scale each plot in this row to the range
            for i in row:
                if i in skipRangeSet:
                    continue
                diff = values[i][1] - values[i][0]
                l2 = (largest - diff) / 2.0
                #self.plotItems[i].getViewBox()._updatingRange = True
                self.plotItems[i].setYRange(values[i][0] - l2, values[i][1] + l2, padding = 0.05)
                
                #self.plotItems[i].getViewBox()._updatingRange = False
                #print(f'{values[i][0] - l2},{values[i][1] + l2}')

    def updateTraceStats(self):
        if self.traceStats:
            self.traceStats.update()

    # color is hex string ie: '#ff0000'
    def startGeneralSelect(self, name, color, timeEdit, canHide=False):
        self.updateLinesAppearance(name, color)
        self.generalSelectStep = 1
        self.generalSelectCanHide = canHide
        self.generalTimeEdit = timeEdit
        self.resetLinesPos('general')
        self.connectLinesToTimeEdit(timeEdit, 'general')

    def connectLinesToTimeEdit(self, timeEdit, lineStr):
        timeEdit.start.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, lineStr))
        timeEdit.end.dateTimeChanged.connect(functools.partial(self.updateLinesByTimeEdit, timeEdit, lineStr))

    def endGeneralSelect(self):
        self.generalSelectStep = 0
        self.generalTimeEdit = None
        self.setLinesVisible(False, 'general')

    def updateLinesByTimeEdit(self, timeEdit, lineStr):
        x0 = self.plotItems[0].getViewBox().lines[lineStr][0].getXPos()
        x1 = self.plotItems[0].getViewBox().lines[lineStr][1].getXPos()
        i0,i1 = self.getTicksFromTimeEdit(timeEdit)
        t0 = self.times[i0]
        t1 = self.times[i1]

        self.updateLinesPos(lineStr, 0, t0 if x0 < x1 else t1)
        self.updateLinesPos(lineStr, 1, t1 if x0 < x1 else t0)

    def updateTimeEditByLines(self, timeEdit, lineStr, index):
        oi = 0 if index == 1 else 1 # i wonder if this works lol, int(not bool(index))
        x = self.plotItems[0].getViewBox().lines[lineStr][index].getXPos()
        ox = self.plotItems[0].getViewBox().lines[lineStr][oi].getXPos()
        t0 = UTCQDate.UTC2QDateTime(FFTIME(x, Epoch=self.epoch).UTC)
        t1 = UTCQDate.UTC2QDateTime(FFTIME(ox, Epoch=self.epoch).UTC)

        timeEdit.setStartNoCallback(min(t0,t1))
        timeEdit.setEndNoCallback(max(t0,t1))

        self.updateLinesPos(lineStr, index, x)

    def updateGeneralLines(self, index, x):
        self.updateLinesPos('general', index, x)
        self.updateTimeEditByLines(self.generalTimeEdit, 'general', index)

    def updateLinesPos(self, lineStr, index, x):
        for pi in self.plotItems:
            pi.getViewBox().lines[lineStr][index].setPos(x)
        self.updateTraceStats()

    def resetLinesPos(self, lineStr):
        self.updateLinesPos(lineStr, 0, self.times[0])
        self.updateLinesPos(lineStr, 1, self.times[self.iiE])

    def setLinesVisible(self, isVisible, lineStr, index=None):
        #print(f'{isVisible} {lineStr} {index} {exc}')
        for pi in self.plotItems:
            vb = pi.getViewBox()
            if index is None:
                for line in vb.lines[lineStr]:
                    line.setVisible(isVisible)
            else:
                vb.lines[lineStr][index].setVisible(isVisible)      

    def matchLinesVisible(self, lineStr):
        for pi in self.plotItems:
            vb = pi.getViewBox()
            vb.lines[lineStr][1].setVisible(vb.lines[lineStr][0].isVisible())

    # gets indices into time array for selected spectra range
    # makes sure first is less than second
    def getTicksFromLines(self):
        sl = self.plotItems[0].getViewBox().lines['general']
        i0 = self.calcTickIndexByTime(sl[0].getXPos())
        i1 = self.calcTickIndexByTime(sl[1].getXPos())
        return (i0,i1) if i0 < i1 else (i1,i0)

    def getTicksFromTimeEdit(self, timeEdit):
        i0 = self.calcTickIndexByTime(FFTIME(UTCQDate.QDateTime2UTC(timeEdit.start.dateTime()), Epoch=self.epoch)._tick)
        i1 = self.calcTickIndexByTime(FFTIME(UTCQDate.QDateTime2UTC(timeEdit.end.dateTime()), Epoch=self.epoch)._tick)
        return (i0,i1) if i0 < i1 else (i1,i0)

    # based on which plots have active spectra lines, return list for each plot of the datastr and pen for each trace
    def getSelectedPlotInfo(self):
        plotInfo = []
        for i,pi in enumerate(self.plotItems):
            if pi.getViewBox().lines['general'][0].isVisible():
                plotInfo.append((self.lastPlotStrings[i], self.plotTracePens[i]))
        return plotInfo

    # show label on topmost line pair
    def updateLineTextPos(self):
        foundFirst = False
        for i,pi in enumerate(self.plotItems):
            lines = pi.getViewBox().lines['general']
            if not foundFirst and lines[0].isVisible():
                lines[0].mylabel.show()
                if lines[1].isVisible():
                    lines[1].mylabel.show()
                foundFirst = True
            else:
                lines[0].mylabel.hide()
                lines[1].mylabel.hide()

    def updateLinesAppearance(self, name, color):
        pen = pg.mkPen(color, width=1, style=QtCore.Qt.DashLine)
        for i,pi, in enumerate(self.plotItems):
            lines = pi.getViewBox().lines['general']
            for line in lines:
                line.setPen(pen)
                line.mylabel.setText(name, color)
        

# look at the source here to see what functions you might want to override or call
#http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/graphicsItems/ViewBox/ViewBox.html#ViewBox
class MagPyViewBox(pg.ViewBox): # custom viewbox event handling
    def __init__(self, window, plotIndex, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        #self.setMouseMode(self.RectMode)
        self.window = window
        self.plotIndex = plotIndex

        generalLines = []
        for i in range(2):
            generalLines.append(LinkedInfiniteLine(functools.partial(window.updateGeneralLines, i), movable=True, angle=90, pos=0, mylabel='GENERAL', labelColor='#000000'))
        self.lines = {'general':generalLines} # incase want to have concurrent line sets
        for key,lines in self.lines.items():
            for line in lines:
                self.addItem(line, ignoreBounds = True)
                line.setBounds((self.window.itO, self.window.itE))
                line.hide()

    def onLeftClick(self, ev):
         # map the mouse click to data coordinates
        mc = self.mapToView(ev.pos())
        x = mc.x()
        y = mc.y()
        #print(f'{x} {y}')

        if self.window.generalSelectStep == 0:
            self.window.openTraceStats(self.plotIndex)
            self.window.startGeneralSelect('STATS', '#009900', self.window.traceStats.ui.timeEdit, True)

        if self.window.generalSelectStep >= 3 and not self.window.getSelectedPlotInfo():
            self.window.generalSelectStep = 1

        if self.window.generalSelectStep > 0 and self.window.generalSelectStep < 3:
            if self.window.generalSelectStep == 1:
                self.window.generalSelectStep += 1
                self.window.setLinesVisible(True, 'general', 0)
                self.window.updateGeneralLines(0,x)
            elif self.window.generalSelectStep == 2:
                self.window.generalSelectStep += 1
                self.window.matchLinesVisible('general')
                self.window.updateGeneralLines(1,x)
                if self.window.edit:
                    Edit.moveToFront(self.window.edit.minVar)

            self.window.updateLineTextPos()
            
        # reselect this plot if it was deselected
        elif self.window.generalSelectStep >= 3 and not self.anyLinesVisible():
            self.SetMyLinesVisible(True)

        self.window.updateTraceStats()


    def anyLinesVisible(self):
        lines = self.lines['general']
        return lines[0].isVisible() or lines[1].isVisible()

    def SetMyLinesVisible(self, isVisible):
        for line in self.lines['general']:
            line.setVisible(isVisible)
        self.window.updateLineTextPos()

    def onRightClick(self, ev):
        if self.window.generalSelectStep > 1 and self.window.generalSelectCanHide: # cancel selection on this plot (if able to)
            self.SetMyLinesVisible(False)
        else:
            pg.ViewBox.mouseClickEvent(self,ev) # default right click

        if not self.window.getSelectedPlotInfo(): # no plots then close
            self.window.closeTraceStats()
        else:
            self.window.updateTraceStats()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.double(): # double clicking will do same as right click
                self.onRightClick(ev)
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


def myexepthook(type, value, tb):
    print(f'{type} {value}')
    traceback.print_tb(tb,limit=3)
    os.system('pause')

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    #appName = f'{appName} {version}';
    app.setOrganizationName('IGPP UCLA')
    app.setOrganizationDomain('igpp.ucla.edu')
    app.setApplicationName('MagPy4')
    #app.setApplicationVersion(version)

    main = MagPy4Window(app)
    main.show()

    sys.excepthook = myexepthook
    sys.exit(app.exec_())