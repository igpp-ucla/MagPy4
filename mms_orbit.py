from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from MagPy4UI import MatrixWidget, VectorWidget, TimeEdit, NumLabel, GridGraphicsLayout, StackedLabel, checkForOrbitLibs
from FF_Time import FFTIME, leapFile
from layoutTools import BaseLayout
from trajectory import OriginGraphic, OrbitPlotter

import os
import sys
import zipfile
from datetime import datetime, timedelta
import numpy as np
import pyqtgraph as pg
import bisect
import json

# Check that libraries are installed before importing
if checkForOrbitLibs():
    import requests
    import cdflib

class MMS_OrbitUI(BaseLayout):
    def setupUI(self, frame):
        frame.setWindowTitle('MMS Orbit')
        wrapLt = QtWidgets.QGridLayout(frame)

        # Get the settings layout/frame
        settingsFrm = QtWidgets.QFrame()
        settingsFrm.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))
        self.settingsLt = self.getSettingsLt(settingsFrm)
        self.settingsLt.setContentsMargins(0, 0, 0, 0)

        # Add in status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))

        # Add settingsFrm to layout above status bar
        layout = QtWidgets.QGridLayout()
        layout.addWidget(settingsFrm, 0, 0, 1, 1)

        # Add sublayout and statusbar to wrapper layout
        wrapLt.addLayout(layout, 0, 0, 1, 1)
        wrapLt.addWidget(self.statusBar, 1, 0, 1, 1)

        # Placeholder for plot grid to be added
        self.gview, self.glw = None, None
        self.layout = layout

    def getSettingsLt(self, frm=None):
        layout = QtWidgets.QVBoxLayout(frm)

        # Set up plot type layout
        plotTypeLt = QtWidgets.QHBoxLayout()
        pltTypeLbl = QtWidgets.QLabel('Plot Type: ')
        self.pltTypeBox = QtWidgets.QComboBox()
        self.pltTypeBox.addItems(['Partial Orbit', 'Full Orbit', 'Multiple Orbits'])
        self.pltTypeBox.currentTextChanged.connect(self.plotTypeChanged)
        for item in [pltTypeLbl, self.pltTypeBox]:
            plotTypeLt.addWidget(item)
        plotTypeLt.addStretch()

        # Set up probe number box
        probeLbl = QtWidgets.QLabel('  Probe #: ')
        self.probeBox = QtWidgets.QComboBox()
        self.probeBox.addItems(['1','2','3','4'])
        probeLt = QtWidgets.QHBoxLayout()
        for item in [probeLbl, self.probeBox]:
            item.setSizePolicy(self.getSizePolicy('Max', 'Max'))
            probeLt.addWidget(item)

        # Set up time edit
        self.timeEdit = TimeEdit(QtGui.QFont())
        self.start = self.timeEdit.start
        self.end = self.timeEdit.end
        for te in [self.start, self.end]:
            te.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        timeLt = QtWidgets.QGridLayout()
        timeLt.setContentsMargins(0, 0, 0, 0)
        self.startLbl = self.addPair(timeLt, 'Start Time: ', self.start, 0, 0, 1, 1)
        self.endLbl = self.addPair(timeLt, 'End Time: ', self.end, 1, 0, 1, 1)

        # Set up axis view
        axisLbl = QtWidgets.QLabel('View Plane: ')
        self.axisBox = QtWidgets.QComboBox()
        self.axisBox.addItems(['XY', 'YZ', 'XZ'])
        axisLt = QtWidgets.QHBoxLayout()
        for item in [axisLbl, self.axisBox]:
            axisLt.addWidget(item)
        axisLt.addLayout(probeLt)

        # Set up options layout
        optionsLt = QtWidgets.QHBoxLayout()
        self.pltRestOrbit = QtWidgets.QCheckBox('Trace Full Orbit')
        self.plotOrigin = QtWidgets.QCheckBox('Plot Origin')
        self.pltRestOrbit.toggled.connect(self.pltOrbitChecked)
        optionsLt.addWidget(self.pltRestOrbit)
        optionsLt.addWidget(self.plotOrigin)
        optionsLt.addStretch()

        # Additional miscellaneous options layout
        miscOptionsLt = QtWidgets.QHBoxLayout()
        self.mmsColorChk = QtWidgets.QCheckBox('Use MMS Colors')
        miscOptionsLt.addWidget(self.mmsColorChk)

        # Set up 'Plot Time Ticks' layout
        self.timeTickFrm = self.getTimeTickFrm()

        # Update button and layout setup
        self.updateBtn = QtWidgets.QPushButton(' Update ')
        self.updateBtn.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Add layouts into grid, wrapping in an hboxlayout w/ a stretch
        # factor at the end to keep the UI elements aligned to the left
        for lt in [plotTypeLt, timeLt, axisLt, optionsLt, miscOptionsLt, self.timeTickFrm]:
            if lt == self.timeTickFrm:
                layout.addWidget(self.timeTickFrm)
                continue

            subLt = QtWidgets.QHBoxLayout()
            subLt.addLayout(lt)
            subLt.addStretch()
            layout.addLayout(subLt)

        layout.addWidget(self.updateBtn, alignment=QtCore.Qt.AlignLeft)
        layout.addStretch()

        return layout

    def getTimeTickFrm(self):
        frame = QtWidgets.QGroupBox(' Plot Time Ticks')
        frame.setCheckable(True)
        frame.setChecked(False)
        lt = QtWidgets.QHBoxLayout(frame)

        # Set up radio buttons
        self.autoBtn = QtWidgets.QRadioButton('Auto')
        self.autoBtn.setChecked(True)
        self.cstmBtn = QtWidgets.QRadioButton('Custom')

        # Set up time box for custom time tick intervals
        self.tickTimeBox = QtWidgets.QTimeEdit()
        self.tickTimeBox.setMinimumDateTime(datetime(2000, 1, 1, 0, 0, 1))
        self.tickTimeBox.setDisplayFormat("HH:mm:ss '(HH:MM:SS)'")

        # Add everything to layout
        lt.addWidget(self.autoBtn)
        lt.addWidget(self.cstmBtn)
        lt.addWidget(self.tickTimeBox)

        # Set minimal size policies
        for btn in [self.autoBtn, self.cstmBtn]:
            btn.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        return frame

    def plotTypeChanged(self):
        # Get new plot type
        plotType = self.pltTypeBox.currentText()

        # Default UI settings
        showRestChk = False # Hide 'Trace Full Orbit' checkbox
        showEndTime = True # Show 'End Time' time edit
        showOriginChk = False # Hide 'Plot Origin' checkbox
        startLbl = 'Start Time: ' # Default label for 'start' time edit
        showTimeTickChk = True

        # If partial orbit plot, show 'Trace Full Orbit' and 'Plot Origin' boxes
        if plotType == 'Partial Orbit':
            showRestChk = True
            showOriginChk = True
        # If full orbit plot, hide 'end' time edit and change 'start' time edit label
        elif plotType == 'Full Orbit':
            showEndTime = False
            startLbl = 'Time: '
        else:
            showTimeTickChk = False

        # Hide/show 'Trace Full Orbit' checkbox
        self.pltRestOrbit.setVisible(showRestChk)

        # Update start/end time edits
        for item in [self.endLbl, self.end]:
            item.setVisible(showEndTime)
        self.startLbl.setText(startLbl)

        # Hide/show 'Plot Origin' checkbox
        self.plotOrigin.setVisible(showOriginChk)
        if plotType != 'Partial Orbit':
            self.plotOrigin.setChecked(True)

        # Hide/show time tick frame/check
        self.timeTickFrm.setVisible(showTimeTickChk)

    def getTimeTickSpacing(self):
        # Gets the time tick interval displayed next to the 'custom' radio
        # button in the 'Plot Time Ticks' layout
        plotType = self.pltTypeBox.currentText()
        timeTicksChk = self.timeTickFrm.isChecked()

        # Returns whether time ticks should be plotted and the spacing
        # if it is specified by the user
        chk = False
        spacing = None
        if timeTicksChk and plotType in ['Partial Orbit', 'Full Orbit']:
            chk = True
            if self.autoBtn.isChecked():
                spacing = None
            else:
                currTime = self.tickTimeBox.dateTime().toPyDateTime()
                spacing = currTime - datetime(2000, 1, 1, 0, 0, 0)

        return (chk, spacing)

    def pltOrbitChecked(self, val):
        # Auto-check 'Plot Origin' box if 'Trace Full Orbit' is checked
        if val:
            self.plotOrigin.setChecked(True)

    def showGrid(self):
        # Initializes grid, plot item, and time info label if it hasn't been
        # shown yet
        if self.glw is None:
            self.glw = self.getGraphicsGrid()
            self.gview.setSizePolicy(self.getSizePolicy('Min', 'Min'))
            self.plt = pg.PlotItem()
            self.timeInfoLbl = pg.LabelItem()
            self.glw.addItem(self.plt, 0, 0, 1, 1)
            self.glw.addItem(self.timeInfoLbl, 1, 0, 1, 1)
            self.layout.addWidget(self.gview, 0, 1, 1, 1)

class MMS_Orbit(QtWidgets.QFrame, MMS_OrbitUI):
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.ui = MMS_OrbitUI()
        self.ui.setupUI(self)
        self.ui.updateBtn.clicked.connect(self.updatePlot)

        # Get min/max datetime in orbit table and set as min/max time edit values
        orbitTool = Orbit_MMS()
        table = orbitTool.readOrbitTable()
        dateLst = table[:,1]
        minDt, maxDt = dateLst[0], dateLst[-1]

        self.ui.timeEdit.setupMinMax((minDt, maxDt))
        minDt, maxDt = window.getMinAndMaxDateTime()
        self.ui.timeEdit.start.setDateTime(minDt)
        self.ui.timeEdit.end.setDateTime(maxDt)

        self.orbitData = {}

        self.axMap = {'X':0, 'Y':1, 'Z':2}

    def updatePlot(self):
        # Extract UI parameters
        plotType, probeNum, timeRng, viewPlane = self.getParameters()

        # Make sure time range is not too large (> 180 days) to work
        # with download limits
        startDt, endDt = timeRng
        diff = abs(endDt-startDt)
        if diff > timedelta(days=180) and plotType != 'Full Orbit':
            self.ui.statusBar.showMessage('Error: Time range too large')
            return
        else:
            self.ui.statusBar.clearMessage()

        # Get checkbox options
        opts = {}
        opts['OrbitTrace'] = self.ui.pltRestOrbit.isChecked()
        opts['MMSColors'] = self.ui.mmsColorChk.isChecked()
        opts['TimeTicks'] = self.ui.getTimeTickSpacing()

        # Plot orbit trace
        self.plotOrbit(plotType, probeNum, timeRng, viewPlane, opts)

        # Plot origin if necessary
        if self.ui.plotOrigin.isChecked() or plotType in ['Full Orbit', 'Multiple Orbits']:
            self.plotOrigin()
    
        # Update plot labels and view range
        units = 'KM'
        self.setPlotLabels(viewPlane, probeNum, units)

        if self.ui.glw:
            self.ui.plt.getViewBox().autoRange()

    def setPlotLabels(self, viewPlane, scNum, units):
        if self.ui.glw is None:
            return

        plt = self.ui.plt
        units = f'({units})' if units != '' else ''

        x_ax, y_ax = viewPlane[0], viewPlane[1]
        y_lbl = f'{y_ax} GSM {units}'
        x_lbl = f'{x_ax} GSM {units}'

        plt.setTitle(f'MMS {scNum} Orbit')
        plt.getAxis('left').setLabel(y_lbl)
        plt.getAxis('bottom').setLabel(x_lbl)

    def plotOrbit(self, plotType, probeNum, timeRng, viewPlane, opts):
        # Adjust time range for full orbit plots since using a single reference time
        startTime, endTime = timeRng
        if plotType == 'Full Orbit':
            timeRng = (startTime, startTime)
        else: # Sort times otherwise
            timeRng = (min(timeRng), max(timeRng))

        # Map view axes to data rows
        y_ax, x_ax = viewPlane[1], viewPlane[0]
        y_axNum, x_axNum = self.axMap[y_ax], self.axMap[x_ax]

        # Get full position data
        times, posDta = self.getOrbitPosDta(probeNum, timeRng)
        if len(times) == 0 or len(posDta) == 0:
            return

        # Get starting/ending indices based on time range
        startTime, endTime = timeRng
        startIndex, endIndex = self.getStartEndIndices(times, startTime, endTime)

        ## Use full time range for full/multiple orbit plots
        if plotType in ['Full Orbit', 'Multiple Orbits']:
            startIndex = 0
            endIndex = len(posDta[0])

        # Select and clip position data
        yDta = posDta[y_axNum][startIndex:endIndex]
        xDta = posDta[x_axNum][startIndex:endIndex]

        # Create plot item if grid is hidden
        self.ui.showGrid()
        self.ui.plt.clear()
        self.ui.plt.hideButtons()
        plt = self.ui.plt

        ## Set aspect ratio
        plt.setAspectLocked(True, 1.0)

        # Get default trace pen, increasing width if also drawing silhouette
        orbitTrace = (opts['OrbitTrace'] and plotType == 'Partial Orbit')

        # Determine which pen color and width to use
        penColor = (25, 127, 255) # Blue
        if opts['MMSColors']:
            penColor = self.window.mmsColors[probeNum-1]

        penWidth = 1.5
        if orbitTrace: # Large width for partial orbit plots w/ silhouette
            penWidth = 2.5
        elif plotType == 'Multiple Orbits': # Regular width for multiple orbits
            penWidth = 1

        pen = pg.mkPen(penColor)
        pen.setWidthF(penWidth)
        pen.setJoinStyle(QtCore.Qt.RoundJoin) # Round edges

        # Plot selected orbit data
        plt.plot(xDta, yDta, pen=pen)

        # Plot optional time ticks
        if opts['TimeTicks'][0]:
            chk, spacing = opts['TimeTicks']
            # Plot ticks along main trace line (partial orbit only for 
            # 'Partial Orbit' plots)
            self.plotTimeTicks(xDta, yDta, times[startIndex:endIndex], pen, spacing)

        # Add starting/ending markers to partial orbit trace
        if plotType == 'Partial Orbit':
            pen = pg.mkPen(pen.color())
            brush = pg.mkBrush((255, 255, 255))
            plt.scatterPlot([xDta[0]], [yDta[0]], symbol='s', pen=pen, size=8, brush=brush)
            plt.scatterPlot([xDta[-1]], [yDta[-1]], symbol='o', pen=pen, size=8, brush=brush)

        # Plot silhouette and set the z-value to be lower than partial orbit trace
        if orbitTrace:
            pen = self.getDashPen()
            item = plt.plot(posDta[x_axNum], posDta[y_axNum], pen=pen)
            z = item.zValue()
            item.setZValue(z-1)

        # Update time info label
        startTick = times[startIndex]
        endTick = times[endIndex-1]
        startUTC = self.window.getTimestampFromTick(startTick)
        endUTC = self.window.getTimestampFromTick(endTick)
        lbl = f'Orbit Time Range: {startUTC} to {endUTC }'
        self.ui.timeInfoLbl.setText(lbl)

        return plt

    def plotTimeTicks(self, xDta, yDta, times, pen, td=None):
        # Check that tick spacing is reasonable before plotting
        if td is not None:
            totSecs = td.total_seconds()
            totDiff = times[-1] - times[0]
            if totDiff / totSecs > 150:
                msg = 'Error: Time tick spacing too small'
                self.ui.statusBar.showMessage(msg)
                return

        # Get time ticks and positions corresponding them
        plt = self.ui.plt
        epoch = 'J2000'
        gaps = np.zeros(len(times)) + 1 # Assume there are no time gaps for now
        res = OrbitPlotter.getTimeTickPositions(xDta, yDta, times, epoch, gaps, td)
        if res is None:
            return

        # Plot tick markers
        pen = pg.mkPen(pen.color())
        brush = pg.mkBrush((255, 255, 255))
        xInterp, yInterp, ticks, axis = res
        plt.scatterPlot(xInterp, yInterp, symbol='o', size=6, pen=pen, brush=brush)

        # Plot time tick labels
        OrbitPlotter.plotTimeTickLabels(plt, (xInterp, yInterp), ticks, axis,
            self.window)

    def getOrbitPosDta(self, probeNum, timeRng):
        # Map start/end time to time ticks
        startTime, endTime = timeRng

        # Open orbit data tool and get the list of required orbits
        orbitTool = Orbit_MMS(timeRng)
        orbitNums = set(orbitTool.getOrbitNums())

        # Find all the saved orbits in the data that do not need to be re-downloaded
        savedOrbits = []
        for (orbit, scNum) in self.orbitData:
            times, data = self.orbitData[(orbit, scNum)]
            if probeNum == scNum and orbit in orbitNums:
                savedOrbits.append(orbit)
        savedOrbits = list(set(savedOrbits))
        savedOrbits.sort()

        # Check if the saved orbits fully cover the time range
        if orbitNums == set(savedOrbits):
            # Concatenate times and position data across all necessary orbits
            times, data = self.mergeOrbitData(savedOrbits, probeNum)
            return times, data

        # Show message in status bar and update UI
        self.ui.statusBar.showMessage(f'Downloading MMS {probeNum} position data...')
        self.ui.processEvents()

        # Download orbit data and add to dictionary
        res = orbitTool.getPosDta(probeNum)

        if res is None:
            self.ui.statusBar.showMessage('Error: Could not download position data')
            return [], []

        orbitNums, timeDict, dataDict = res
        for orbit in orbitNums:
            orbitDta = dataDict[orbit]
            orbitTimes = timeDict[orbit]
            self.orbitData[(orbit, probeNum)] = (orbitTimes, orbitDta)

        # Clear status bar and update UI
        self.ui.statusBar.clearMessage()
        self.ui.processEvents()

        # Concatenate orbit times and data now that they are in the dictionary
        times, data = self.mergeOrbitData(orbitNums, probeNum)

        return times, data

    def mergeOrbitData(self, orbitNums, scNum):
        # Concatenate the times and position data (3xN format) for each orbit
        times = []
        data = [[],[],[]]
        for orbit in orbitNums:
            if (orbit, scNum) in self.orbitData:
                orbitTimes, orbitDta = self.orbitData[(orbit, scNum)]
                times = np.concatenate([times, orbitTimes])
                data = np.concatenate([data, orbitDta], axis=1)

        return times, data

    def getStartEndTimes(self):
        # Get start/end datetime objects and make sure they are sorted
        startTime = self.ui.start.dateTime().toPyDateTime()
        endTime = self.ui.end.dateTime().toPyDateTime()
        return (startTime, endTime)

    def getStartEndTicks(self, startTime, endTime):
        # Map start/end datetimes to seconds since epoch
        startTick = self.window.getTickFromDateTime(startTime)
        endTick = self.window.getTickFromDateTime(endTime)
        return (startTick, endTick)

    def getStartEndIndices(self, times, startTime, endTime):
        # Find indices corresponding to start/end ticks in the times array
        startTick, endTick = self.getStartEndTicks(startTime, endTime)
        startIndex = bisect.bisect_left(times, startTick)
        endIndex = bisect.bisect(times, endTick)
        return (startIndex, endIndex)

    def plotOrigin(self):
        if self.ui.glw is None:
            return
        pen = pg.mkPen((0, 0, 0))
        origin = OriginGraphic(radius=6371.2, origin=(0, 0), pen=pen)
        self.ui.plt.addItem(origin)

    def getParameters(self):
        # Extract probe number, time range, and view plane from UI
        plotType = self.ui.pltTypeBox.currentText()
        probeNum = int(self.ui.probeBox.currentText())

        timeRng = self.getStartEndTimes()

        viewPlane = self.ui.axisBox.currentText()
        return (plotType, probeNum, timeRng, viewPlane)

    def getDashPen(self, color='#000000'):
        # Pen used for tracing full orbit silhouette in partial orbit plots
        pen = pg.mkPen(color)
        pen.setDashPattern([3, 4])
        pen.setWidthF(1.5)
        return pen

class Orbit_MMS():
    '''
        Finds orbit number corresponding to given time range and downloads
        position data from LASP server
    '''
    def __init__(self, timeRange=None):
        self.timeFmt = '%Y-%m-%dT%H:%M:%S.%f'
        self.ttDenom = 10 ** 9
        self.setTimeRange(timeRange)

    def setTimeRange(self, timeRange):
        if timeRange is not None:
            self.timeRange = timeRange
            self.orbitNum, self.fullRange, self.orbitTimes = self.getOrbitNumber(timeRange)
        else:
            self.timeRange = None
            self.orbitNum = []
            self.fullRange = []
            self.orbitTimes = []

    def getOrbitNums(self):
        # Returns the list of orbit numbers within the given time range
        return self.orbitNum

    def getFullRange(self):
        # Returns the start/end time covering all the orbits listed
        # in self.orbitNum
        return self.fullRange
    
    def getOrbitTimes(self):
        # List of start/end times for each individual orbit in self.orbitNums
        return self.orbitTimes

    def readOrbitTable(self):
        # Open orbit table
        fd = open('orbittable.txt', 'r')

        # Get the start/end indices corresponding to the orbit number and perigee times
        orbitHeader = 'ORBIT NUMBER  '
        perigeeHeader = 'PERIGEE TIME             '
        i1 = len(orbitHeader)
        i2 = i1 + len(perigeeHeader)

        # Read lines, convert times to datetimes, and cast table to a numpy array
        lines = fd.readlines()[1:]
        mapTsToDt = lambda s : self.mapTimestampToDt(s, self.timeFmt)
        lines = [[int(line[:i1]), mapTsToDt(line[i1:i2].strip(' '))] for line in lines]
        table = np.array(lines)
        return table

    def getOrbitNumber(self, timeRange):
        # Read in the orbit table
        table = self.readOrbitTable()
        orbitNums, perigeeTimes = table[:,0], table[:,1]

        # Find bins that start and end times fall in
        startTime, endTime = timeRange

        startIndex = bisect.bisect(perigeeTimes, startTime)
        startIndex = max(startIndex - 1, 0)

        endIndex = bisect.bisect_left(perigeeTimes, endTime)
        endIndex = min(endIndex, len(perigeeTimes) - 1)

        # Extract orbit numbers and the start/end time range
        orbitNum = orbitNums[startIndex:endIndex]
        orbitRange = (perigeeTimes[startIndex], perigeeTimes[endIndex])

        ## Get the time ranges for each specific orbit
        orbitRanges = []
        for i in range(startIndex, endIndex):
            subRange = (perigeeTimes[i], perigeeTimes[i+1])
            orbitRanges.append(subRange)

        return orbitNum, orbitRange, orbitRanges

    def getQueryURL(self, scNum, orbitRng):
        # Default query parameters
        base_url = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/file_info/science?'
        rate = 'data_rate_mode=srvy'
        level = 'data_level=l2'
        instr = 'instrument_id=mec'

        # Specify spacecraft number
        sc = f'sc_ids=mms{scNum}'

        # Specify start and end date
        fmtDt = lambda d: d.strftime('%Y-%m-%d')
        startDt, endDt = orbitRng
        startDt = f'start_date={fmtDt(startDt)}'
        endDt = f'end_date={fmtDt(endDt)}'

        url = '&'.join([base_url, instr, sc, rate, level, startDt, endDt])
        return url

    def downloadFiles(self, files):
        # Form the download query from the list of files and download zip file
        base_url = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?files='
        file_query = ','.join(files)
        url = base_url + file_query
        res = requests.get(url)

        # Write response contents to a zip folder
        fd = open('files.zip', 'wb')
        fd.write(res.content)
        fd.close()
        return 'files.zip'

    def mapTimestampToDt(self, s, fmt):
        return datetime.strptime(s, fmt)

    def getFiles(self, scNum, orbitRng):
        url = self.getQueryURL(scNum, orbitRng)

        # Query server and convert to dictionary
        try:
            res = requests.get(url)
        except: # Returns None if could not establish a connection
            return None 
        res = json.loads(res.text)

        # Download files as zip
        files = [f['file_name'] for f in res['files'] if 'epht89d' in f['file_name']]
        fileName = self.downloadFiles(files)

        # Unzip files
        with zipfile.ZipFile(fileName, 'r') as f:
            f.extractall()
        f.close()

        # Sort the list of filenames by start date
        mapTsToDt = lambda s : self.mapTimestampToDt(s, self.timeFmt[:-3])
        resFiles = []
        resTimestamps = []
        for d in res['files']:
            if d['file_name'] not in files:
                continue
            resFiles.append(d['file_name'])
            resTimestamps.append(mapTsToDt(d['timetag']))

        sortMask = np.argsort(resTimestamps)

        return [resFiles[i] for i in sortMask]

    def getPosDta(self, scNum):
        # Download and get the list of files within the orbit range for given 
        # spacecraft number
        orbitRng = self.getFullRange()
        files = self.getFiles(scNum, orbitRng)

        # Return None if URL request failed (no connection)
        if files is None: 
            return None

        paths = ['']*len(files)

        # Files are split among different subdirectories, get bottom-most
        # directories and set path for the corresponding file
        for dirpath, dirnames, filenames in os.walk('mms'):
            if dirnames:
                continue
            for f in filenames:
                if f not in files:
                    continue
                index = files.index(f)
                paths[index] = os.path.join(dirpath, f)

        # Read in data from CDFs into dicts and extract position data
        timeKw = 'Epoch'
        kw = f'mms{scNum}_mec_r_gsm'
        times = []
        dateTimes = []
        posDta = [[],[],[]]
        for f in paths:
            d = self.readCDF(f)

            # Concatenate times
            cdfTimes = d[timeKw]
            times = np.concatenate([times, cdfTimes])

            # Concatenate position data (3xn format)
            cdfPosDta = np.array(d[kw]).T
            posDta = np.concatenate([posDta, cdfPosDta], axis=1)

            # Remove files after done reading
            os.remove(f)

        # Store times and data for each orbit in dictionaries w/ the key being
        # the orbit number
        dataDict = {}
        timeDict = {}
        base = (10 ** 9)
        for orbitNum, (startTime, endTime) in zip(self.orbitNum, self.orbitTimes):
            # Map datetime objects to seconds since epoch
            startTime = self.mapTTNanoseconds(self.dateTimeToTT(startTime))
            endTime = self.mapTTNanoseconds(self.dateTimeToTT(endTime))

            # Find indices corresponding to orbit time range
            startIndex = bisect.bisect(times, startTime)
            endIndex = bisect.bisect(times, endTime)

            # Store data between start/end indices in dictionaries
            dataDict[orbitNum] = posDta[:,startIndex:endIndex+1]
            timeDict[orbitNum] = times[startIndex:endIndex+1]

        return self.orbitNum, timeDict, dataDict

    def dateTimeToTT(self, dt):
        # Convert datetime to a list of its individual date/time components
        # and pass that to cdfepoch to convert to TT2000 time
        lst = [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]
        msStr = str(dt.microsecond)
        milliseconds = int(msStr[:3])
        microseconds = int(msStr[3:])
        nanoseconds = 0
        lst = lst + [milliseconds, microseconds, nanoseconds]
        return cdflib.cdfepoch.compute_tt2000(lst)

    def mapTTNanoseconds(self, t):
        # Converts TT2000 time in nanoseconds to seconds since J2000
        dt = 32.184
        return (t / self.ttDenom) - dt

    def readCDF(self, fn):
        # Open CDF
        cdf = cdflib.CDF(fn)
        d = {}

        # Create a dictionary of data
        timeKw = 'Epoch'
        kw = f'_mec_r_gsm'
        varNames = cdf.cdf_info()['zVariables']
        for v in varNames:
            if kw in v or timeKw in v:
                d[v] = cdf.varget(v)

        # Map epoch times
        d['Epoch'] = self.mapTTNanoseconds(d['Epoch'])

        return d