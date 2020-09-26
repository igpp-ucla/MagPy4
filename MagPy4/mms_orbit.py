from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from .plotBase import MagPyPlotItem
from .MagPy4UI import MatrixWidget, VectorWidget, TimeEdit, NumLabel, GridGraphicsLayout, StackedLabel, checkForOrbitLibs
from .layoutTools import BaseLayout
from .trajectory import OriginGraphic, OrbitPlotter, MagnetosphereTool
from . import getRelPath

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

        # Set up coordinate system combo box
        coordLt = QtWidgets.QHBoxLayout()
        coordLbl = QtWidgets.QLabel('Coordinate System: ')
        coordLbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.coordBox = QtWidgets.QComboBox()
        self.coordBox.addItems(['GSM', 'GSE', 'SM'])
        coordLt.addWidget(coordLbl)
        coordLt.addWidget(self.coordBox)

        # Set up scaling mode box
        scaleLt = QtWidgets.QHBoxLayout()
        scaleLbl = QtWidgets.QLabel('Scale: ')
        scaleLbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.scaleBox = QtWidgets.QComboBox()
        self.scaleBox.addItems(['Kilometers', 'Earth Radii'])
        for item in [scaleLbl, self.scaleBox]:
            scaleLt.addWidget(item)

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

        # Set up magnetosphere model button/layout
        self.modelFrm = self.getMagnetModelFrm()

        # Update button and layout setup
        self.updateBtn = QtWidgets.QPushButton(' Update ')
        self.updateBtn.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Add layouts into grid, wrapping in an hboxlayout w/ a stretch
        # factor at the end to keep the UI elements aligned to the left
        settingsFrm = QtWidgets.QGroupBox('Settings')
        settingsLt = QtWidgets.QVBoxLayout(settingsFrm)
        for lt in [plotTypeLt, timeLt, axisLt, coordLt, scaleLt, optionsLt,
            miscOptionsLt]:
            subLt = QtWidgets.QHBoxLayout()
            subLt.addLayout(lt)
            subLt.addStretch()
            settingsLt.addLayout(subLt)
        layout.addWidget(settingsFrm)

        # Add time tick and magnetosphere model options frames
        layout.addWidget(self.timeTickFrm)
        layout.addWidget(self.modelFrm)

        layout.addWidget(self.updateBtn, alignment=QtCore.Qt.AlignLeft)
        layout.addStretch()

        return layout

    def getTimeTickFrm(self):
        frame = QtWidgets.QGroupBox(' Plot Time Ticks')
        frame.setCheckable(True)
        frame.setChecked(False)
        lt = QtWidgets.QGridLayout(frame)

        # Set up radio buttons
        self.autoBtn = QtWidgets.QRadioButton('Auto')
        self.autoBtn.setChecked(True)
        self.cstmBtn = QtWidgets.QRadioButton('Custom')

        # Set up time box for custom time tick intervals
        self.tickTimeBox = QtWidgets.QTimeEdit()
        self.tickTimeBox.setMinimumDateTime(datetime(2000, 1, 1, 0, 0, 1))
        self.tickTimeBox.setDisplayFormat("HH:mm:ss '(HH:MM:SS)'")

        # Add everything to layout
        lt.addWidget(self.autoBtn, 0, 0, 1, 1)
        lt.addWidget(self.cstmBtn, 1, 0, 1, 1)
        lt.addWidget(self.tickTimeBox, 1, 1, 1, 1)

        # Set minimal size policies
        for btn in [self.autoBtn, self.cstmBtn]:
            btn.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        return frame
    
    def getMagnetModelFrm(self):
        frm = QtWidgets.QGroupBox(' Plot Magnetosphere Model')
        frm.setCheckable(True)
        frm.setChecked(False)
        frm.toggled.connect(self.plotModelChecked)

        # Button for accessing model parameters widget
        self.modelBtn = QtWidgets.QPushButton('Set Model Parameters')
        self.modelBtn.setMaximumWidth(250)

        # Add single button to layout
        lt = QtWidgets.QGridLayout(frm)
        lt.addWidget(self.modelBtn, 0, 0, 1, 1)

        return frm

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

    def plotModelChecked(self, val):
        # Auto-check 'Plot Origin' box if 'Plot Magnetosphere Model' is checked
        if val:
            self.plotOrigin.setChecked(True)

    def showGrid(self):
        # Initializes grid, plot item, and time info label if it hasn't been
        # shown yet
        if self.glw is None:
            self.glw = self.getGraphicsGrid()
            self.gview.setSizePolicy(self.getSizePolicy('Min', 'Min'))
            self.plt = MagPyPlotItem()
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
        self.ui.modelBtn.clicked.connect(self.openMagModelTool)

        # Keeps track of magnetosphere model tool
        self.magModelTool = None

        # Update model coordinate system if main coord system changed
        self.ui.coordBox.currentTextChanged.connect(self.updateModelCoordSys)

        # Earth radius in km
        self.earthRadius = 6371.2

        # Get min/max datetime in orbit table and set as min/max time edit values
        orbitTool = Orbit_MMS()
        table = orbitTool.readOrbitTable()
        dateLst = table[:,1]
        minDt, maxDt = dateLst[0], dateLst[-1]

        self.ui.timeEdit.setupMinMax((minDt, maxDt))
        minDt, maxDt = window.getMinAndMaxDateTime()
        self.ui.timeEdit.start.setDateTime(minDt)
        self.ui.timeEdit.end.setDateTime(maxDt)

        # Set up dictionaries for storing orbit data and mapping axes to rows
        self.orbitData = {}
        self.axMap = {'X':0, 'Y':1, 'Z':2}

    def updatePlot(self):
        # Extract UI parameters
        plotType, scNum, timeRng, viewPlane, coordSys = self.getParameters()

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
        opts['Scale'] = self.ui.scaleBox.currentText()

        # Plot orbit trace
        self.plotOrbit(plotType, scNum, timeRng, viewPlane, coordSys, opts)

        # Plot origin if necessary
        if self.ui.plotOrigin.isChecked() or plotType in ['Full Orbit', 'Multiple Orbits']:
            self.plotOrigin(opts['Scale'])

        # Plot magnetosphere model if checked
        if self.ui.modelFrm.isChecked():
            self.plotMagnetosphere(viewPlane, coordSys, opts['Scale'])
    
        # Update plot labels and view range
        units = 'KM' if opts['Scale'] == 'Kilometers' else 'RE'
        self.setPlotLabels(viewPlane, scNum, units, coordSys)

        if self.ui.glw:
            QtCore.QTimer.singleShot(100, self.ui.plt.getViewBox().autoRange)

    def setPlotLabels(self, viewPlane, scNum, units, coordSys):
        if self.ui.glw is None:
            return

        plt = self.ui.plt
        units = f'({units})' if units != '' else ''

        x_ax, y_ax = viewPlane[0], viewPlane[1]
        y_lbl = f'{y_ax} {coordSys.upper()} {units}'
        x_lbl = f'{x_ax} {coordSys.upper()} {units}'

        plt.setTitle(f'MMS {scNum} Orbit')
        plt.getAxis('left').setLabel(y_lbl)
        plt.getAxis('bottom').setLabel(x_lbl)

    def plotOrbit(self, plotType, probeNum, timeRng, viewPlane, coordSys, opts):
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
        times, posDta = self.getOrbitPosDta(probeNum, timeRng, coordSys)
        if len(times) == 0 or len(posDta) == 0:
            return

        # Scale position data if necessary
        if opts['Scale'] == 'Earth Radii':
            posDta = posDta / self.earthRadius

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

    def getOrbitPosDta(self, probeNum, timeRng, coordSys):
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
            times, data = self.mergeOrbitData(savedOrbits, probeNum, coordSys)
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
            # posDta is a dictionary of the orbit data in the different
            # coordinate systems
            posDta = {}
            for kw in ['gsm', 'gse', 'sm']:
                posDta[kw] = dataDict[(orbit, kw)]
            orbitTimes = timeDict[orbit]
            # orbitData stores the orbit data described above, and
            # the corresponding time ticks, with the keys being the 
            # orbit number and the probe number
            self.orbitData[(orbit, probeNum)] = (orbitTimes, posDta)

        # Clear status bar and update UI
        self.ui.statusBar.clearMessage()
        self.ui.processEvents()

        # Concatenate orbit times and data now that they are in the dictionary
        times, data = self.mergeOrbitData(orbitNums, probeNum, coordSys)

        return times, data

    def mergeOrbitData(self, orbitNums, scNum, coordSys='gsm'):
        # Concatenate the times and position data (3xN format) for each orbit
        times = []
        data = [[],[],[]]
        for orbit in orbitNums:
            if (orbit, scNum) in self.orbitData:
                orbitTimes, orbitDta = self.orbitData[(orbit, scNum)]
                # Extract the data for the right coordinate system
                orbitDta = orbitDta[coordSys]
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

    def getEarthRadius(self):
        return self.earthRadius

    def getMinMaxDt(self):
        return self.window.getMinAndMaxDateTime()

    def plotOrigin(self, scale='Kilometers'):
        if self.ui.glw is None:
            return
        # Determine which radius to use based on scale mode
        if scale != 'Kilometers':
            radius = 1.0
        else:
            radius = self.earthRadius

        # Add graphic with given radius
        pen = pg.mkPen((0, 0, 0))
        origin = OriginGraphic(radius=radius, origin=(0, 0), pen=pen)
        self.ui.plt.addItem(origin)

    def getParameters(self):
        # Extract probe number, time range, and view plane from UI
        plotType = self.ui.pltTypeBox.currentText()
        probeNum = int(self.ui.probeBox.currentText())

        timeRng = self.getStartEndTimes()

        viewPlane = self.ui.axisBox.currentText()
        coordSys = self.ui.coordBox.currentText().lower()

        return (plotType, probeNum, timeRng, viewPlane, coordSys)

    def getDashPen(self, color='#000000'):
        # Pen used for tracing full orbit silhouette in partial orbit plots
        pen = pg.mkPen(color)
        pen.setDashPattern([3, 4])
        pen.setWidthF(1.5)
        return pen

    def openMagModelTool(self):
        self.initMagModelTool()
        self.magModelTool.show()

    def initMagModelTool(self):
        if self.magModelTool is None:
            self.magModelTool = MagnetosphereTool(self)

            # Set default coordinate system to selected coordinate system
            coordSys = self.ui.coordBox.currentText()
            self.magModelTool.ui.coordBox.setCurrentText(coordSys)

    def updateModelCoordSys(self, txt):
        if self.magModelTool:
            self.magModelTool.ui.coordBox.setCurrentText(txt)

    def plotMagnetosphere(self, viewPlane, coordSys, scale='Kilometers'):
        # Open magnetosphere tool if it hasn't been opened
        self.initMagModelTool()

        # Map view axes to data rows
        y_ax, x_ax = viewPlane[1], viewPlane[0]
        y_axNum, x_axNum = self.axMap[y_ax], self.axMap[x_ax]

        # Update status bar to
        self.ui.statusBar.showMessage('Calculating magnetosphere field line coordinates...')
        self.ui.processEvents()

        # Calculate field line coordinates and plot each line
        xDta, yDta, tiltAngle = self.magModelTool.getFieldLines(y_axNum, x_axNum)
        pen = pg.mkPen('ff910d')

        ## Scale coordinates to RE if selected by user
        if scale != 'Kilometers':
            xDta = np.array(xDta) / self.earthRadius
            yDta = np.array(yDta) / self.earthRadius

        for x, y in zip(xDta, yDta):
            item = self.ui.plt.plot(x, y, pen=pen)
            # Set z value so it's beneath origin graphic and orbit lines
            z = item.zValue()
            item.setZValue(z-3)

        self.ui.statusBar.clearMessage()

    def closeEvent(self, ev):
        if self.magModelTool:
            self.magModelTool.close()
            self.magModelTool = None
        self.close()

class MMS_Data_Tool():
    '''
    Downloads position data for a given time range from the LASP MMS Science 
    Data Center website
    '''
    ttDenom = 10 ** 9
    timeFmt = '%Y-%m-%dT%H:%M:%S.%f'

    def getQueryURL(scNums, orbitRng):
        # Default query parameters
        base_url = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/file_info/science?'
        rate = 'data_rate_mode=srvy'
        level = 'data_level=l2'
        instr = 'instrument_id=mec'

        # Specify spacecraft number(s)
        scIds = ','.join([f'mms{scNum}' for scNum in scNums])
        sc = f'sc_ids={scIds}'

        # Specify start and end date
        fmtDt = lambda d: d.strftime('%Y-%m-%d')
        startDt, endDt = orbitRng
        startDt = f'start_date={fmtDt(startDt)}'
        endDt = f'end_date={fmtDt(endDt)}'

        url = '&'.join([base_url, instr, sc, rate, level, startDt, endDt])
        return url

    def downloadFiles(files):
        # Form the download query from the list of files and download zip file
        base_url = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?files='
        file_query = ','.join(files)
        url = base_url + file_query
        res = requests.get(url)

        # Write response contents to a zip folder
        zipName = 'files.zip'
        zipPath = getRelPath(zipName)
        fd = open(zipPath, 'wb')
        fd.write(res.content)
        fd.close()
        return zipPath

    def mapTimestampToDt(s, fmt):
        return datetime.strptime(s, fmt)

    def getFiles(scNums, orbitRng):
        url = MMS_Data_Tool.getQueryURL(scNums, orbitRng)

        # Query server and convert to dictionary
        try:
            res = requests.get(url)
        except: # Returns None if could not establish a connection
            return None 
        res = json.loads(res.text)

        # Download files as zip
        files = [f['file_name'] for f in res['files'] if 'epht89d' in f['file_name']]
        fileName = MMS_Data_Tool.downloadFiles(files)

        # Unzip files
        with zipfile.ZipFile(fileName, 'r') as f:
            f.extractall(path=getRelPath())
        f.close()

        # Sort the list of filenames by start date
        timeFmt = MMS_Data_Tool.timeFmt
        mapTsToDt = lambda s : MMS_Data_Tool.mapTimestampToDt(s, timeFmt[:-3])
        resFiles = []
        resTimestamps = []
        resSpacecrafts = [int(f[3]) for f in files]
        for d in res['files']:
            if d['file_name'] not in files:
                continue
            resFiles.append(d['file_name'])
            resTimestamps.append(mapTsToDt(d['timetag']))

        sortMask = np.argsort(resTimestamps)

        return [resFiles[i] for i in sortMask], resSpacecrafts

    def getPosDta(timeRng, scNums):
        # Download and get the list of files within the orbit range for given 
        # spacecraft number
        files, fileScNums = MMS_Data_Tool.getFiles(scNums, timeRng)

        # Return None if URL request failed (no connection)
        if files is None: 
            return None

        paths = ['']*len(files)

        # Files are split among different subdirectories, get bottom-most
        # directories and set path for the corresponding file
        directories = []
        mms_zip_path = getRelPath('mms')
        for dirpath, dirnames, filenames in os.walk(mms_zip_path):
            if dirnames:
                continue

            filesFound = False
            for f in filenames:
                if f not in files:
                    continue
                filesFound = True
                index = files.index(f)
                paths[index] = os.path.join(dirpath, f)

            if filesFound:
                directories.append(dirpath)

        # Read in data from CDFs into dicts and extract position data
        timeKw = 'Epoch'
        scTimes = {scNum:[] for scNum in scNums}
        coordKws = ['gsm', 'gse', 'sm']
        scDta = {scNum:{coordKw:[[],[],[]] for coordKw in coordKws} for scNum in scNums}
        for f, fScNum in zip(paths, fileScNums):
            kw = f'mms{fScNum}_mec_r_'
            d = MMS_Data_Tool.readCDF(f)

            # Concatenate times
            cdfTimes = d[timeKw]
            scTimes[fScNum] = np.concatenate([scTimes[fScNum], cdfTimes])

            # Concatenate position data (3xn format)
            for coordKw in coordKws:
                cdfPosDta = np.array(d[kw+coordKw]).T
                oldPosDta = scDta[fScNum][coordKw]
                newPosDta = np.concatenate([oldPosDta, cdfPosDta], axis=1)
                scDta[fScNum][coordKw] = newPosDta

            # Remove files after done reading
            os.remove(f)

        # Remove the unzipped file folders
        try:
            for path in directories:
                os.removedirs(path)
        except:
            print ('Error: Could not remove temporarily downloaded files')

        return scDta, scTimes

    def readCDF(fn):
        # Open CDF
        cdf = cdflib.CDF(fn)
        d = {}

        # Create a dictionary of data
        timeKw = 'Epoch'
        kw = f'_mec_r_'
        varNames = cdf.cdf_info()['zVariables']
        for v in varNames:
            if kw in v or timeKw in v:
                d[v] = cdf.varget(v)

        # Map epoch times
        d['Epoch'] = MMS_Data_Tool.mapTTNanoseconds(d['Epoch'])

        return d

    def mapTTNanoseconds(t):
        # Converts TT2000 time in nanoseconds to seconds since J2000
        dt = 32.184
        return (t / MMS_Data_Tool.ttDenom) - dt

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
        orbitTablePath = getRelPath('orbittable.txt')
        fd = open(orbitTablePath, 'r')

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

    def mapTimestampToDt(self, s, fmt):
        return datetime.strptime(s, fmt)

    def getPosDta(self, scNum):
        # Download and get the list of files within the orbit range for given 
        # spacecraft number
        orbitRng = self.getFullRange()
        scDta, scTimes = MMS_Data_Tool.getPosDta(orbitRng, [scNum])

        # Return None if URL request failed (no connection)
        if scDta is None: 
            return None

        # Extract position data for this specific spacecraft
        posDta = scDta[scNum]
        times = scTimes[scNum]

        # Store times and data for each orbit in dictionaries w/ the key being
        # the orbit number
        dataDict = {}
        timeDict = {}
        coordKws = ['gsm', 'gse', 'sm']
        for orbitNum, (startTime, endTime) in zip(self.orbitNum, self.orbitTimes):
            # Map datetime objects to seconds since epoch
            startTime = self.mapTTNanoseconds(self.dateTimeToTT(startTime))
            endTime = self.mapTTNanoseconds(self.dateTimeToTT(endTime))

            # Find indices corresponding to orbit time range
            startIndex = bisect.bisect(times, startTime)
            endIndex = bisect.bisect(times, endTime)

            # Store data between start/end indices in dictionaries
            for coordKw in coordKws:
                dataDict[(orbitNum, coordKw)] = posDta[coordKw][:,startIndex:endIndex+1]
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

