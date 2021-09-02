from PyQt5 import QtGui, QtCore, QtWidgets
from .MagPy4UI import TimeEdit
from PyQt5.QtWidgets import QSizePolicy
from .layouttools import BaseLayout
from .plotbase import DateAxis, MagPyPlotItem
from .dynbase import GradLegend, ColorBar
import pyqtgraph as pg
from pyqtgraph import GraphicsWidgetAnchor
import numpy as np
from scipy import interpolate
from .mth import Mth

# Magnetosphere modules
import sys
from .geopack.geopack import geopack
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing import Pool
from fflib import ff_time

class TrajectoryUI(BaseLayout):
    def setupUI(self, Frame, window):
        Frame.setWindowTitle('Trajectory Analysis')
        Frame.resize(1100, 800)

        layout = QtWidgets.QGridLayout(Frame)
        fieldVecs, posVecs = Frame.fieldVecs, Frame.posVecs
        settingsFrm = self.setupSettingsFrame(fieldVecs, posVecs, window)
        self.glw = self.getGraphicsGrid(window)

        # Create tabs and tab widget
        self.altFrame = AltitudePlotter(Frame)
        self.orbitFrame = OrbitPlotter(Frame)

        self.tabWidget = QtWidgets.QTabWidget()
        self.tabWidget.addTab(self.altFrame, 'Altitude')
        self.tabWidget.addTab(self.orbitFrame, 'Orbit')

        timeLt, self.timeEdit, self.statusBar = self.getTimeStatusBar()

        layout.addWidget(settingsFrm, 0, 0, 1, 3)
        layout.addWidget(self.tabWidget, 1, 0, 1, 3)
        layout.addLayout(timeLt, 2, 0, 1, 3)

    def setupSettingsFrame(self, fieldVecs, posVecs, window):
        frame = QtWidgets.QFrame()
        frame.setSizePolicy(self.getSizePolicy('Min', 'Max'))
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(2, 0, 2, 0)

        # Setup combo boxes for field and position vectors
        self.posBoxes = []
        self.vecBoxes = []
        row = 0
        for lbl, vecGrps in zip(['Field Vector:', 'Position Vector: '],
            [fieldVecs, posVecs]):
            # Set up label
            lbl = QtWidgets.QLabel(lbl)
            layout.addWidget(lbl, row, 0, 1, 1)

            # Setup boxes and fill them in
            for axis in range(0, 3):
                box = QtWidgets.QComboBox()
                if lbl.text() == 'Field Vector:':
                    self.vecBoxes.append(box)
                else:
                    self.posBoxes.append(box)
                for grp in vecGrps:
                    box.addItem(grp[axis])
                layout.addWidget(box, row, axis+1, 1, 1)
            row += 1

        # Radius box
        self.radiusBox = QtWidgets.QDoubleSpinBox()
        self.radiusBox.setMaximum(1e9)
        self.radiusBox.setMinimum(0.0001)
        self.radiusBox.setValue(1)
        self.radiusBox.setDecimals(7)
        self.addPair(layout, ' Radius: ', self.radiusBox, 0, 4, 1, 1)

        # Radius units
        self.radUnitBox = QtWidgets.QLineEdit()
        self.radUnitBox.setMaxLength(15)
        defaultUnit = window.UNITDICT[posVecs[0][-1]]
        if len(defaultUnit) > 0 and 'R'.lower() in defaultUnit[0].lower():
            defaultUnit = defaultUnit.upper()
        if 'rti' in defaultUnit.lower():
            defaultUnit = 'RTI'
        self.radUnitBox.setText(defaultUnit)
        self.addPair(layout, ' Units: ', self.radUnitBox, 1, 4, 1, 1)

        for row in range(0, 2):
            maxPol, minPol = QSizePolicy.Maximum, QSizePolicy.MinimumExpanding
            spacer = QtWidgets.QSpacerItem(5, 1, minPol, maxPol)
            layout.addItem(spacer, row, 6, 1, 1)

        return frame

class TrajectoryAnalysis(QtWidgets.QFrame, TrajectoryUI):
    def __init__(self, window, parent=None):
        super().__init__(parent)
        self.window = window
        self.epoch = window.epoch
        self.ui = TrajectoryUI()

        self.fieldVecs, self.posVecs = self.getFieldAndPosVecs()

        if not self.validState():
            return

        self.ui.setupUI(self, window)
        self.ui.timeEdit.setupMinMax(window.getMinAndMaxDateTime())

    def update(self):
        self.ui.altFrame.updatePlot()
        self.ui.orbitFrame.updatePlot()

    def getState(self):
        generalState = {}
        generalState['Vecs'] = (self.getFieldVec(), self.getPosVec())
        generalState['Radius'] = (self.getRadius(), self.getRadiusUnits())
        generalState['Altitude'] = self.ui.altFrame.getState()
        generalState['Orbit'] = self.ui.orbitFrame.getState()
        return generalState

    def loadState(self, state):
        # Load selected position and field vectors
        fieldVec, posVec = state['Vecs']
        for i, fdstr, pdstr in zip([0,1,2], fieldVec, posVec):
            self.ui.vecBoxes[i].setCurrentText(fdstr)
            self.ui.posBoxes[i].setCurrentText(pdstr)

        # Load specified radius and units
        radius, units = state['Radius']
        self.ui.radUnitBox.setText(units)
        self.ui.radiusBox.setValue(radius)

        # Load tool data
        self.ui.altFrame.loadState(state['Altitude'])
        self.ui.orbitFrame.loadState(state['Orbit'])

    def validState(self):
        # Checks if at least one field vec and pos vec could be identified
        minLength = (len(self.fieldVecs) > 0 and len(self.posVecs) > 0)
        if not minLength:
            return False

        # Check that lengths of times for field vectors
        # and position vectors are the same
        fieldDstr = self.fieldVecs[0]
        posDstr = self.posVecs[0]
        fieldTimes = self.getTimes(fieldDstr[0], 0)
        posTimes = self.getTimes(posDstr[-1], 0)
        if len(fieldTimes) != len(posTimes):
            return False
        
        return True

    def getSegments(self, sI, eI, vec='pos'):
        # Finds the time gap indices and returns a list indicating
        # connections between points for use w/ pg's plot function
        # so time gaps aren't connected
        en = self.getEditNum()
        vec = self.getPosVec() if vec == 'pos' else self.getFieldVec()
        times, resolutions, avgRes = self.window.getTimes(vec[0], en)
        resolutions = resolutions[sI:eI-1]
        mask = resolutions > (avgRes * 2)
        segments = np.array(np.logical_not(mask), dtype=np.int32)
        segments = np.concatenate([segments, [0]])
        return segments

    def getFieldVec(self):
        return [box.currentText() for box in self.ui.vecBoxes]

    def getPosVec(self):
        return [box.currentText() for box in self.ui.posBoxes]

    def getTimestampFromTick(self, t):
        return self.window.getTimestampFromTick(t)

    def getIndices(self):
        dstr = self.getFieldVec()[0]
        a, b = self.window.calcDataIndicesFromLines(dstr, 0)
        return (a, b)

    def getData(self, dstr, en):
        return self.window.getData(dstr, en)

    def getTimes(self, dstr, en):
        return self.window.getTimes(dstr, en)[0]

    def getPens(self):
        return self.window.pens

    def getRadius(self):
        return self.ui.radiusBox.value()

    def getEarthRadius(self):
        if self.getRadiusUnits().lower() == 'km':
            return 6371.2
        else:
            return 1

    def getRadiusUnits(self):
        return self.ui.radUnitBox.text()

    def getEditNum(self):
        return self.window.currentEdit

    def getFieldAndPosVecs(self):
        fieldVecs = []
        posVecs = []
        fieldKws = ['Bx', 'By', 'Bz']
        posKws = ['X', 'Y', 'Z']

        dstrs = self.window.DATASTRINGS[:]

        # Find groups of var names matching keywords
        grps = []
        for kw in fieldKws:
            grp = []
            for dstr in dstrs:
                if kw.lower() in dstr.lower():
                    grp.append(dstr)
            grps.append(grp)

        # Find max number of fully-included vectors
        grpLens = list(map(len, grps))
        minLen = min(grpLens)

        # Group kw groups into vector groups
        for vecNum in range(0, minLen):
            fieldVecs.append([grp[vecNum] for grp in grps])

        if minLen < 1:
            fieldVecs.append(self.window.DATASTRINGS[0:3])

        # Find position vectors similar to above
        grps = []
        for kw in posKws:
            grp = []
            for dstr in dstrs:
                if kw.lower() == dstr[0].lower():
                    grp.append(dstr)
            grps.append(grp)

        grpLens = list(map(len, grps))
        minLen = min(grpLens)

        for vecNum in range(0, minLen):
            posVecs.append([grp[vecNum] for grp in grps])

        return fieldVecs, posVecs

    def closeEvent(self, event):
        self.window.endGeneralSelect()
        self.close()
        if self.validState():
            self.ui.orbitFrame.closeMagTool()

class AltitudeUI(BaseLayout):
    def setupUI(self, Frame, outerFrame):
        layout = QtWidgets.QGridLayout(Frame)
        layout.setContentsMargins(0, 4, 0, 0)
        settingsFrame = self.setupSettingsFrame(Frame, outerFrame)
        self.glw = self.getGraphicsGrid()
        self.glw.setContentsMargins(4, 0, 4, 6)
        layout.addWidget(settingsFrame, 0, 0, 1, 1)
        layout.addWidget(self.gview, 1, 0, 1, 1)

    def setupSettingsFrame(self, Frame, outerFrame):
        frame = QtWidgets.QFrame()
        frame.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        layout = QtWidgets.QGridLayout(frame)

        # Variable box
        self.dstrBox = QtWidgets.QComboBox()
        self.dstrBox.addItems(Frame.plotTypes)
        self.addPair(layout, 'Plot Type: ', self.dstrBox, 0, 0, 1, 1)

        spacer = self.getSpacer(10)
        layout.addItem(spacer, 0, 2, 1, 1)

        # Link checkbox
        self.linkBox = QtWidgets.QCheckBox('Link X Axes')
        self.linkBox.setChecked(True)
        layout.addWidget(self.linkBox, 0, 3, 1, 1)

        # Update button
        self.updtBtn = QtWidgets.QPushButton('Update')
        layout.addWidget(self.updtBtn, 0, 4, 1, 1)

        return frame

class AltitudePlotter(QtWidgets.QFrame, AltitudeUI):
    def __init__(self, outerFrame, parent=None):
        super().__init__(parent)
        self.outerFrame = outerFrame
        self.epoch = outerFrame.epoch
        self.plots = []
        self.plotTypes = ['Bx, By, Bz', 'Cone & Clock Angles', 'Bt']

        self.ui = AltitudeUI()
        self.ui.setupUI(self, outerFrame)
        self.ui.linkBox.clicked.connect(self.linkAxes)
        self.ui.updtBtn.clicked.connect(self.updatePlot)

    def linkChecked(self):
        return self.ui.linkBox.isChecked()

    def getState(self):
        plotType = self.getPlotType()
        linkedAxes = self.linkChecked()
        return (plotType, linkedAxes)

    def loadState(self, state):
        plotType, linkedAxes = state
        self.ui.dstrBox.setCurrentText(plotType)
        self.ui.linkBox.setChecked(linkedAxes)

    def linkAxes(self, val=None):
        if val is None:
            val = self.linkChecked()

        if val:
            xDta = []
            for plt in self.plots:
                plt.enableAutoRange(x=False)
                pdi = plt.listDataItems()[0]
                xDta.append(pdi.xData)

            # Find the max view range and average value for each plot
            avgVals = []
            maxDiff = None
            for dta in xDta:
                minVal = np.min(dta)
                maxVal = np.max(dta)
                if np.isnan(minVal) or np.isnan(maxVal):
                    avgVals.append(None)

                diff = maxVal - minVal
                if maxDiff is None or diff > maxDiff:
                    maxDiff = diff

                avgVal = (minVal + maxVal) / 2
                avgVals.append(avgVal)

            if maxDiff == None:
                return

            # Update plots to have same x scale, centered around the avg value
            diff = maxDiff / 2
            p = diff * 0.05 # Strict padding
            for plt, avgVal in zip(self.plots, avgVals):
                if avgVal is None:
                    continue
                plt.setXRange(avgVal-diff-p, avgVal+diff+p, 0)

        else:
            for plt in self.plots:
                plt.enableAutoRange(x=True)

    def getPlotType(self):
        return self.ui.dstrBox.currentText()

    def calcAltitude(self, posDstrs, a, b, radius, mask):
        # Computes r = sqrt(x^2+y^2+z^2), then (r-1) * radius
        en = self.outerFrame.getEditNum()
        posDta = [self.outerFrame.getData(dstr, en)[a:b][mask] for dstr in posDstrs]

        # Mask out any error flag data points
        rDta = np.sqrt((posDta[0] ** 2) + (posDta[1] ** 2) + (posDta[2] ** 2))

        alt = (rDta - 1) * radius

        return alt

    def getVecData(self, dstrs, en, a, b, mask):
        vecDta = []
        for dstr in dstrs:
            fieldDta = self.outerFrame.getData(dstr, en)[a:b]
            vecDta.append(fieldDta[mask])

        return vecDta

    def calcMagDta(self, dstrs, en, a, b, mask):
        dta = self.getVecData(dstrs, en, a, b, mask)
        magDta = np.sqrt((dta[0]**2)+(dta[1]**2)+(dta[2]**2))
        return magDta

    def calcConeAngle(self, dstrs, en, a, b, mask):
        # arccos(Bx / Bmag)
        bxDstr = dstrs[0]
        bxDta = self.outerFrame.getData(bxDstr, en)[a:b][mask]
        magDta = self.calcMagDta(dstrs, en, a, b, mask)
        coneAngle = np.arccos(bxDta/magDta)
        return (coneAngle * Mth.R2D)

    def calcClockAngle(self, dstrs, en, a, b, mask):
        # arctan(Bz / By)
        byDstr = dstrs[1]
        bzDstr = dstrs[2]
        byDta = self.outerFrame.getData(byDstr, en)[a:b]
        bzDta = self.outerFrame.getData(bzDstr, en)[a:b]
        clockAngle = np.arctan(bzDta/byDta) * Mth.R2D
        return clockAngle[mask]

    def getTimeLbl(self, times):
        t0, t1 = times[0], times[-1]
        startTime = self.outerFrame.getTimestampFromTick(t0)
        endTime = self.outerFrame.getTimestampFromTick(t1)
        lbl = 'Time Range: ' + startTime + ' to ' + endTime
        lbl = pg.LabelItem(lbl)
        lbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        return lbl

    def getErrorMask(self, fieldDstrs, vecDstrs, en, a, b):
        dstrs = fieldDstrs + vecDstrs
        mask = []
        for dstr in dstrs:
            dta = self.outerFrame.getData(dstr, en)[a:b]
            if mask == []:
                mask = dta < self.outerFrame.window.errorFlag
            else:
                mask = mask & (dta < self.outerFrame.window.errorFlag)

        return mask

    def updatePlot(self):
        # Extract user-selections from UI
        en = self.outerFrame.getEditNum()
        a, b = self.outerFrame.getIndices()
        plotType = self.getPlotType()
        radius = self.outerFrame.getRadius()
        radiiUnits = self.outerFrame.getRadiusUnits()
        posDstrs = self.outerFrame.getPosVec()

        # Generate altitude and selected data
        dstrs = self.outerFrame.getFieldVec()
        mask = self.getErrorMask(dstrs, posDstrs, en, a, b)
        altDta = self.calcAltitude(posDstrs, a, b, radius, mask)
        xUnit = 'nT'
        if plotType == 'Bx, By, Bz':
            fieldDtaLst = self.getVecData(dstrs, en, a, b, mask)
            lbls = dstrs
        elif plotType == 'Bt':
            fieldDtaLst = [self.calcMagDta(dstrs, en, a, b, mask)]
            lbls = ['Bt']
        elif plotType == 'Cone & Clock Angles':
            fieldDtaLst = [self.calcConeAngle(dstrs, en, a, b, mask)]
            fieldDtaLst.append(self.calcClockAngle(dstrs, en, a, b,mask))
            lbls = ['Cone Angle', 'Clock Angle']
            xUnit = 'Degrees'

        # Create a plot for each dataset in fieldDtaLst
        plts = []
        index = 0
        for fieldDta, dstr in zip(fieldDtaLst, lbls):
            # Create plot item
            plt = MagPyPlotItem()
            pen = self.outerFrame.getPens()[index]

            gaps = self.outerFrame.getSegments(a, b)
            gaps = gaps[mask]
            plt.plot(fieldDta, altDta, pen=pen, connect=gaps)
            plt.hideButtons()

            # Set plot labels
            altLbl = 'Altitude ('+radiiUnits+')' if radiiUnits != '' else 'Altitude'
            plt.getAxis('left').setLabel(altLbl)
            plt.getAxis('bottom').setLabel(dstr + ' (' + xUnit + ')')
            plt.setTitle('Altitude Profile')
            plts.append(plt)
            index += 1

        # Clear previous grid and add plots to sublayout
        self.ui.glw.clear()
        pltLt = pg.GraphicsLayout() # Sublayout for plot items
        for col in range(0, len(plts)):
            plt = plts[col]
            pltLt.addItem(plt, 0, col, 1, 1)
            plt.setSizePolicy(self.getSizePolicy('Min', 'Min'))

        # Add time info
        times = self.outerFrame.getTimes(posDstrs[0], en)
        timeLbl = self.getTimeLbl(times)

        # Add stretch and spacers
        if len(plts) < 3:
            col = len(plts)
            stretch = 2 if col == 2 else 1
            for prevCol in range(0, col):
                pltLt.layout.setColumnStretchFactor(prevCol, stretch)

            spacer = self.getGraphicSpacer('Min', 'Min')
            pltLt.addItem(spacer, 0, col, 1, 1)
            pltLt.layout.setColumnStretchFactor(col, 1)
    
        # Save current plots and update links between plot axes
        self.plots = plts
        for z in range(1, len(self.plots)): # Link y axes for scaling purposes
            v1 = self.plots[z-1].getViewBox()
            v2 = self.plots[z].getViewBox()
            v1.setYLink(v2)
        self.linkAxes()

        # Add plot layout and time label to main layout
        self.ui.glw.addItem(pltLt, 0, 0, 1, 1)
        self.ui.glw.addItem(timeLbl, 1, 0, 1, 1)

class OrbitUI(BaseLayout):
    def setupUI(self, Frame, outerFrame):
        self.outerFrame = outerFrame
        layout = QtWidgets.QGridLayout(Frame)
        settingsFrame = self.setupSettingsFrame(Frame, outerFrame)
        self.glw = self.getGraphicsGrid()
        self.glw.setContentsMargins(5, 0, 5, 0)
        layout.addWidget(settingsFrame, 0, 0, 1, 1)

        spacer = QtWidgets.QSpacerItem(1, 1, QSizePolicy.Maximum, QSizePolicy.Expanding)
        layout.addItem(spacer, 1, 0, 1, 1)

        layout.addWidget(self.gview, 0, 1, 2, 1)
    
    def setupSettingsFrame(self, Frame, outerFrame):
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(4, 4, 4, 4)
        frame.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Plot type frame
        pltFrm = QtWidgets.QGroupBox('Plot Type:')
        pltLt = QtWidgets.QGridLayout(pltFrm)

        # Plane of view radio button and axis boxes
        self.viewPlaneBtn = QtWidgets.QRadioButton('View Plane: ')
        self.viewPlaneBtn.setChecked(True)
        self.axesList = ['X','Y','Z']
        self.planeBox1 = QtWidgets.QComboBox()
        self.planeBox2 = QtWidgets.QComboBox()
        for kw in self.axesList:
            self.planeBox1.addItem(kw)
            self.planeBox2.addItem(kw)
        self.planeBox1.addItem('YZ')
        self.planeBox1.setCurrentIndex(1)
        self.planeBox1.currentTextChanged.connect(self.updateBoxItems)
        pltLt.addWidget(self.viewPlaneBtn, 0, 0, 1, 1)
        pltLt.addWidget(self.planeBox1, 0, 1, 1, 1)
    
        self.addPair(pltLt, ' by ', self.planeBox2, 0, 2, 1, 1)

        # Projection onto the terminator plane radio button
        self.projBtn = QtWidgets.QRadioButton('Projection onto Terminator Plane')
        self.projBtn.setToolTip('Projection of field lines onto the terminator plane')
        pltLt.addWidget(self.projBtn, 1, 0, 1, 4)

        layout.addWidget(pltFrm, 0, 0, 1, 5)

        # Build tick settings sub-layouts and group into a single layout
        tickSettingsLt = QtWidgets.QVBoxLayout()
        tickSettingsLt.setContentsMargins(0,0,0,0)
        tickSettingsLt.setSpacing(10)

        self.fieldLineBox = self.setupFieldLineLt()
        self.timeTickBox = self.setupTimeLt()
        self.projLt = self.setupProjLt()

        for frm in [self.fieldLineBox, self.timeTickBox]:
            tickSettingsLt.addWidget(frm)
        tickSettingsLt.addLayout(self.projLt)
    
        layout.addLayout(tickSettingsLt, 2, 0, 1, 5)

        # Magnetosphere model group box
        modelFrm = self.setupModelLt(outerFrame, Frame)
        layout.addWidget(modelFrm, 4, 0, 1, 5)

        # Additional plot settings layout
        pltLt = self.setupPlotSettingsLt(outerFrame, Frame)
        layout.addLayout(pltLt, 6, 0, 1, 5)

        # Add in cosmetic separators
        for row in [1, 3,5,7]:
            self.addSeparator(layout, row, 5)

        # Update button
        self.updtBtn = QtWidgets.QPushButton('Update')
        layout.addWidget(self.updtBtn, 87, 4, 1, 1)
        return frame

    def addSeparator(self, layout, row, span):
        separator = self.getWidgetSeparator()
        layout.addWidget(separator, row, 0, 1, span, QtCore.Qt.AlignVCenter)

    def updateBoxItems(self):
        # Makes sure if 'YZ' is selected, than the other axis must be 'X'
        txt = self.planeBox1.currentText()
        if txt == 'YZ':
            self.planeBox2.clear()
            self.planeBox2.addItem('X')
        elif self.planeBox2.count() != len(self.axesList):
            self.planeBox2.clear()
            self.planeBox2.addItems(self.axesList)

    def getWidgetSeparator(self):
        # Line separator between layout elements
        spacer = QtWidgets.QFrame()
        spacer.setFrameStyle(QtWidgets.QFrame.HLine)
        spacer.setMinimumHeight(15)
        spacer.setStyleSheet('color: rgb(191, 191, 191);')
        return spacer

    def setupPlotSettingsLt(self, outerFrame, Frame):
        layout = QtWidgets.QGridLayout()

        # Plot title
        titleLt = QtWidgets.QGridLayout()
        self.plotTitleBox = QtWidgets.QLineEdit()
        self.plotTitleBox.setText('Orbit Plot')
        self.plotTitleBox.setMaxLength(72)
        self.addPair(titleLt, 'Plot Title: ', self.plotTitleBox, 0, 0, 1, 1)
        layout.addLayout(titleLt, 0, 0, 1, 2)

        # Lock aspect ratio check
        self.aspectBox = QtWidgets.QCheckBox('1:1 Aspect Ratio')
        self.aspectBox.setChecked(True)
        layout.addWidget(self.aspectBox, 1, 0, 1, 1)

        # Plot origin check
        self.pltOriginBox = QtWidgets.QCheckBox('Plot Origin')
        self.pltOriginBox.setChecked(False)
        layout.addWidget(self.pltOriginBox, 1, 1, 1, 1)

        return layout

    def setupFieldLineLt(self):
        frame = QtWidgets.QGroupBox(' Plot Field Lines: ')
        frame.setCheckable(True)
        layout = QtWidgets.QGridLayout(frame)
        lm, tm, rm, bm = layout.getContentsMargins()
        layout.setContentsMargins(lm+9, 5, rm, 4)

        # Tick interval box
        intLt = QtWidgets.QGridLayout()
        self.intervalBox = QtWidgets.QSpinBox()
        self.intervalBox.setMinimum(1)
        self.intervalBox.setMaximum(100000)
        self.intervalBox.setSuffix(' pts')
        self.intervalBox.setSingleStep(5)
        tt = 'Number of points between time ticks/field lines'
        lbl = self.addPair(layout, 'Tick Spacing:  ', self.intervalBox, 0, 0, 1, 1, tt)

        layout.addLayout(intLt, 1, 0, 1, 3)
        # Field line scale
        scaleLt = QtWidgets.QGridLayout()
        self.magLineScaleBox = QtWidgets.QDoubleSpinBox()
        self.magLineScaleBox.setMinimum(1e-5)
        self.magLineScaleBox.setMaximum(1e8)
        self.magLineScaleBox.setDecimals(5)
        self.magLineScaleBox.setValue(0.25)
        self.magLineScaleBox.setMaximumWidth(200)
        tt = 'Ratio between field values and position data units'
        lbl = self.addPair(layout, 'Field Line Scale:  ', self.magLineScaleBox, 1, 0, 1, 1, tt)

        # Center field lines check
        tt = 'Toggle to center field vectors around the orbit line'
        self.centerLinesBox = QtWidgets.QCheckBox('Center Field Vectors')
        self.centerLinesBox.setChecked(True)
        self.centerLinesBox.setToolTip(tt)
        layout.addWidget(self.centerLinesBox, 3, 0, 1, 2, QtCore.Qt.AlignLeft)

        return frame

    def setupTimeLt(self):
        frame = QtWidgets.QGroupBox(' Plot Time Ticks: ')
        frame.setCheckable(True)
        layout = QtWidgets.QGridLayout(frame)
        lm, tm, rm, bm = layout.getContentsMargins()
        layout.setContentsMargins(lm+9, 5, rm, 4)

        # Spacing mode radio buttons
        lbl = QtWidgets.QLabel('Tick Spacing: ')
        self.autoBtn = QtWidgets.QRadioButton('Auto')
        self.customBtn = QtWidgets.QRadioButton('Custom')
        self.autoBtn.setChecked(True)

        # Time interval edit
        self.timeBox = QtWidgets.QTimeEdit()
        self.timeBox.setDisplayFormat("HH:mm:ss '(HH:MM:SS)'")
        self.timeBox.setMinimumTime(QtCore.QTime(0, 0, 1))
        
        layout.addWidget(lbl, 0, 0, 1, 4)
        layout.addWidget(self.autoBtn, 1, 0, 1, 1)
        layout.addWidget(self.customBtn, 1, 1, 1, 1)
        layout.addWidget(self.timeBox, 1, 2, 1, 1)

        return frame

    def setupProjLt(self):
        layout = QtWidgets.QGridLayout()
        layout.setVerticalSpacing(2)
        layout.setContentsMargins(0,0,0,0)

        # Tick intervals box
        self.projIntBox = QtWidgets.QSpinBox()
        self.projIntBox.setMinimum(1)
        self.projIntBox.setMaximum(100000)
        self.projIntBox.setSuffix(' pts')
        self.projIntBox.setSingleStep(5)
        lbl = self.addPair(layout, 'Tick Spacing: ', self.projIntBox, 0, 0, 1, 1)
        self.projIntLbl = lbl

        # Color-mapped checkbox
        self.colorMapBox = QtWidgets.QCheckBox('Color-mapped')
        self.colorMapBox.setToolTip('Map each point to a color based on its time')
        layout.addWidget(self.colorMapBox, 1, 0, 1, 2, QtCore.Qt.AlignLeft)

        return layout

    def setupModelLt(self, outerFrame, Frame):
        self.modelBox = QtWidgets.QGroupBox(' Plot Model of Earth\'s Magnetosphere')
        layout = QtWidgets.QHBoxLayout(self.modelBox)
        lm, tm, rm, bm = layout.getContentsMargins()
        layout.setContentsMargins(lm+9, 5, rm, 4)
        self.modelBox.setCheckable(True)
        self.modelBox.setChecked(False)
        self.modelBox.toggled.connect(self.modelChecked)
        self.modelBtn = QtWidgets.QPushButton('Set Model Parameters')
        self.modelBtn.setFixedWidth(230)

        layout.addWidget(self.modelBtn, QtCore.Qt.AlignLeft)
        layout.addStretch()
        return self.modelBox

    def modelChecked(self, val):
        # Toggle 'Plot Origin' if model is to be plotted
        # and remove 'YZ' from the view axes options if val is True
        numItems = self.planeBox1.count()
        boxItems = [self.planeBox1.itemText(index) for index in range(numItems)]
        if val:
            self.pltOriginBox.blockSignals(True)
            self.pltOriginBox.setChecked(True)
            self.pltOriginBox.blockSignals(False)
            if 'YZ' in boxItems:
                index = boxItems.index('YZ')
                self.planeBox1.removeItem(index)
        elif 'YZ' not in boxItems:
            self.planeBox1.addItem('YZ')

class OrbitPlotter(QtWidgets.QFrame, OrbitUI):
    def __init__(self, outerFrame, parent=None):
        super().__init__(parent)
        self.outerFrame = outerFrame
        self.ui = OrbitUI()

        self.magTool = None
        self.currPlot = None
        self.originItem = None
        self.scaleSet = None
        self.axisKwNum = {'X':0, 'Y':1, 'Z':2, 'YZ':-1}
        self.prevScale = None
        self.earthRadius = 1

        self.ui.setupUI(self, outerFrame)

        # UI connections
        self.ui.updtBtn.clicked.connect(self.updatePlot)
        self.ui.aspectBox.clicked.connect(self.lockAspect)
        self.ui.plotTitleBox.textChanged.connect(self.setPltTitle)
        self.ui.pltOriginBox.toggled.connect(self.addOriginItem)

        self.ui.modelBtn.clicked.connect(self.openMagTool)
        self.outerFrame.ui.radUnitBox.textChanged.connect(self.updtUnits)
        self.outerFrame.ui.radiusBox.valueChanged.connect(self.adjustScale)
        self.ui.magLineScaleBox.valueChanged.connect(self.saveScale)

        # Initialize the units label for mag line scaling
        self.updtUnits(self.outerFrame.getRadiusUnits())
        self.plotTypeChanged()

        # Enable/disable relavent plot options when tick and plot types are changed
        for btn in [self.ui.projBtn, self.ui.viewPlaneBtn]:
            btn.toggled.connect(self.plotTypeChanged)

    def getState(self):
        state = {}

        # General settings
        origin = self.ui.pltOriginBox.isChecked()
        title = self.ui.plotTitleBox.text()
        unitRatio = self.ui.aspectBox.isChecked()
        state['General'] = (origin, title, unitRatio)
        state['MagnetoModel'] = self.getMagnetosphereState()

        # Plot type / view axes if not a terminator plane proj plot
        projMode = self.inProjMode()
        if projMode:
            state['viewAxes'] = None
            state['tickSpacing'] = self.ui.projIntBox.value()
            state['colorMap'] = self.ui.colorMapBox.isChecked()
            return state
        else:
            ax_a = self.ui.planeBox1.currentText()
            ax_b = self.ui.planeBox2.currentText()
            state['viewAxes'] = (ax_a, ax_b)

            # Field line parameters
            fieldLines = self.ui.fieldLineBox.isChecked()
            if fieldLines:
                spacing = self.ui.intervalBox.value()
                scale = self.ui.magLineScaleBox.value()
                center = self.ui.centerLinesBox.isChecked()
                state['Field'] = (spacing, scale, center)
            else:
                state['Field'] = None

            # Time tick parameters (interval value for 'auto' mode is None)
            timeTicks = self.ui.timeTickBox.isChecked()
            state['Time'] = (timeTicks, self.getTimeInterval())

        return state

    def loadState(self, state):
        self.initValues()
        axes = state['viewAxes']

        # Set spacing + color map box for projection-type plots
        if axes is None:
            self.ui.projBtn.setChecked(True)
            spacing, colorMap = state['tickSpacing'], state['colorMap']
            self.ui.projIntBox.setValue(spacing)
            self.ui.colorMapBox.setChecked(colorMap)
        else:
            # Set default view axes
            ax_a, ax_b = axes
            self.ui.planeBox1.setCurrentText(ax_a)
            self.ui.planeBox2.setCurrentText(ax_b)

            # Set field line parameters
            if state['Field'] is not None:
                self.ui.fieldLineBox.setChecked(True)
                spacing, scale, center = state['Field']
                self.ui.intervalBox.setValue(spacing)
                self.ui.magLineScaleBox.setValue(scale)
                self.ui.centerLinesBox.setChecked(center)
            else:
                self.ui.fieldLineBox.setChecked(False)

            # Time tick parameters
            timeTicks, timeInterval = state['Time']
            if timeTicks:
                if timeInterval is None:
                    self.ui.autoBtn.setChecked(True)
                else:
                    self.ui.customBtn.setChecked(True)
                    dt = QtWidgets.QTimeEdit().minimumDateTime().toPyDateTime() 
                    dt = dt + timeInterval
                    self.ui.timeBox.setDateTime(dt)
            else:
                self.ui.timeTickBox.setChecked(False)

        # Additional general plot parameters
        origin, title, unitRatio = state['General']
        self.ui.pltOriginBox.setChecked(origin)
        self.ui.aspectBox.setChecked(unitRatio)
        if title != '':
            self.ui.plotTitleBox.blockSignals(True)
            self.ui.plotTitleBox.setText(title)
            self.ui.plotTitleBox.blockSignals(False)

        # Load magnetosphere model settings and state
        magnetoState = state['MagnetoModel']
        if magnetoState:
            val, modelState = magnetoState
            self.ui.modelBox.setChecked(val)

            # Initialize magnetosphere tool and set model params state
            self.initMagTool()
            self.magTool.loadState(modelState)

    def getMagnetosphereState(self):
        if self.magTool is None:
            return None
        else:
            check = self.ui.modelBox.isChecked()
            modelState = self.magTool.getState()
            return (check, modelState)

    def openMagTool(self):
        if self.magTool:
            self.magTool.show()
        else:
            self.initMagTool()
            self.magTool.show()
        self.magTool.updtTime()

    def closeMagTool(self):
        if self.magTool:
            self.magTool.close()

    def initMagTool(self):
        self.magTool = MagnetosphereTool(self)

    def initValues(self):
        a, b = self.outerFrame.getIndices()
        self.updateTickInterval()
        posDstrs = self.outerFrame.getPosVec()
        vecDstrs = self.outerFrame.getFieldVec()
        self.setVecScale(posDstrs, vecDstrs, a,b)
        self.scaleSet = True

    def getTickType(self):
        tickType = 'None'
        if self.ui.fieldLineBox.isChecked():
            tickType = 'Field Lines'

        if self.ui.timeTickBox.isChecked():
            if tickType != 'None':
                tickType = 'Both'
            else:
                tickType = 'Time Ticks'
        return tickType

    def getScalingFactor(self):
        scaleVal = self.ui.magLineScaleBox.value()
        return scaleVal

    def getOriginRadius(self, posDstrs):
        gsmGseFound = False
        for dstr in posDstrs:
            if 'gsm' in dstr.lower() or 'gse' in dstr.lower():
                gsmGseFound = True
                break

        if gsmGseFound and self.outerFrame.getRadiusUnits().lower() == 'km':
            return 6371.2 / self.outerFrame.getRadius() # Radius of earth in km
        else:
            return self.outerFrame.getRadius()

    def getPrevScale(self):
        # Returns the previously set magnetic field scale / previous radius
        return self.prevScale

    def adjustScale(self, val):
        # Adjusts the magnetic field line scale when the radius is changed
        scale = self.getPrevScale()
        self.ui.magLineScaleBox.setValue(scale/self.outerFrame.getRadius())

    def saveScale(self, val):
        self.prevScale = val * self.outerFrame.getRadius()

    def setPltTitle(self, valStr):
        self.currPlot.setTitle(valStr)

    def updtUnits(self, valStr):
        # Update the units label in the field line scale box when
        # units are changed
        if valStr == '':
            self.ui.magLineScaleBox.setSuffix('')
        else:
            self.ui.magLineScaleBox.setSuffix(' (nT / ' + valStr +'^2)')

        if valStr.lower() == 'km':
            self.earthRadius = 6371.2
        else:
            self.earthRadius = 1

    def lockAspect(self, val):
        if self.currPlot:
            if val:
                self.currPlot.setAspectLocked(True, ratio=1)
            else:
                self.currPlot.setAspectLocked(False)
            self.currPlot.hideScaleBar(not val)

    def inProjMode(self):
        return self.ui.projBtn.isChecked()

    def plotTypeChanged(self):
        # Disable all tick options except for tick interval if plotting a
        # projection of the field lines
        projMode = self.inProjMode()
        if projMode:
            # Update interval box value
            a, b = self.outerFrame.getIndices()
            self.ui.projIntBox.setValue(max(1, int((b-a)*.002)))
        else:
            self.updateTickInterval()
        
        self.ui.timeTickBox.setVisible(not projMode)
        self.ui.fieldLineBox.setVisible(not projMode)
        self.ui.colorMapBox.setChecked(projMode)

        for item in [self.ui.colorMapBox, self.ui.projIntBox, self.ui.projIntLbl]:
            item.setVisible(projMode)

    def updateTickInterval(self):
        # Enable/disable UI elements according to tick type chosen
        # Update tick spacing if switching between marker types
        a, b = self.outerFrame.getIndices()
        val = int((b-a)/75)

        if val is not None:
            val = max(val - (val % 5), 1)
            self.ui.intervalBox.setValue(val)

    def setVecScale(self, posDstrs, vecDstrs, sI, eI):
        en = self.outerFrame.getEditNum()
        posDta = [self.outerFrame.getData(dstr, en)[sI:eI] for dstr in posDstrs]
        vecDta = [self.outerFrame.getData(dstr, en)[sI:eI] for dstr in vecDstrs]

        # Mask out error flags
        mask = [True] * len(posDta[0])
        allDta = list(posDta) + list(vecDta)
        for dta in allDta:
            mask = mask & (dta < self.outerFrame.window.errorFlag)
        posDta = [dta[mask] for dta in posDta]
        vecDta = [dta[mask] for dta in vecDta]

        # Calculate max distance in position
        posStart = [np.max(dta) for dta in posDta]
        posEnd = [np.min(dta) for dta in posDta]
        posDiff = np.array(posStart) - np.array(posEnd)
        totDist = np.sqrt(np.dot(posDiff, posDiff))

        # Calculate median vector magnitude
        vecMags = self.calcMagDta(vecDta)
        avgMag = np.median(vecMags)

        # Scale is set to (max distance / 8) / (median vec length)
        magLen = (totDist / 8) / avgMag
        self.ui.magLineScaleBox.setValue(1/magLen)

    def calcMagDta(self, dta):
        magDta = np.sqrt((dta[0]**2)+(dta[1]**2)+(dta[2]**2))
        return magDta

    def getYZDta(self, posDstrs, vecDstrs, sI, eI, en):
        fieldDta = [self.outerFrame.getData(dstr, en)[sI:eI] for dstr in vecDstrs]
        posDta = [self.outerFrame.getData(dstr, en)[sI:eI] for dstr in posDstrs]

        xField, yField, zField = fieldDta
        xDta, yDta, zDta = posDta

        posYZ = np.sqrt((yDta ** 2) + (zDta ** 2))
        fieldYZ = (yField*yDta + zField*zDta)/posYZ

        return xDta, posYZ, xField, fieldYZ

    def getProjTermDta(self, posDstrs, vecDstrs, sI, eI, en):
        fieldDta = [self.outerFrame.getData(dstr, en)[sI:eI] for dstr in vecDstrs]
        posDta = [self.outerFrame.getData(dstr, en)[sI:eI] for dstr in posDstrs]

        xField, yField, zField = fieldDta
        xDta, yDta, zDta = posDta

        xVals = yDta - (xDta*yField/xField)
        yVals = zDta - (xDta*zField/xField)

        return xVals, yVals
    
    def getViewAxes(self):
        return (self.getAxisNum(1), self.getAxisNum(2))

    def getPosAndField(self, posDstrs, vecDstrs, sI, eI):
        # Get the plot's x,y coordinates in the plane chosen by user
        en = self.outerFrame.getEditNum()
        aAxisNum, bAxisNum = self.getViewAxes()

        if aAxisNum < 0:
            yzDta = self.getYZDta(posDstrs, vecDstrs, sI, eI, en)
            xDta, yDta, xField, yField = yzDta
        else:
            aDstr = posDstrs[aAxisNum]
            bDstr = posDstrs[bAxisNum]

            yDta = self.outerFrame.getData(aDstr, en)[sI:eI] # Limit data to selection
            xDta = self.outerFrame.getData(bDstr, en)[sI:eI]

            aDstr = vecDstrs[aAxisNum]
            bDstr = vecDstrs[bAxisNum]

            yField = self.outerFrame.getData(aDstr, en)[sI:eI]
            xField = self.outerFrame.getData(bDstr, en)[sI:eI]

        return xDta, yDta, xField, yField

    def getAxisNum(self, axisLabelNum=1):
        # Maps axis letter to axis number
        if axisLabelNum == 1:
            axis = self.ui.planeBox1.currentText()
        else:
            axis = self.ui.planeBox2.currentText()

        return self.axisKwNum[axis]

    def sampleData(self, dta, times, rate):
        return dta[::rate]

    def getMaxWidth(self, xDta, yDta):
        # Magnitude between max/min values of position data
        yDiff = max(yDta) - min(yDta)
        xDiff = max(xDta) - min(xDta)
        return max(abs(yDiff), abs(xDiff))

    def addTimeInfo(self, t0, t1):
        startTime = self.outerFrame.getTimestampFromTick(t0)
        endTime = self.outerFrame.getTimestampFromTick(t1)
        lbl = 'Time Range: ' + startTime + ' to ' + endTime
        lbl = pg.LabelItem(lbl)
        self.ui.glw.addItem(lbl, 1, 0, 1, 1)
        lbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.ui.glw.layout.setAlignment(lbl, QtCore.Qt.AlignLeft)

    def getOutlineColors(self, colors):
        # Reduce RGB values by 50 to get a slightly darker color
        outlineColors = []
        width = 50
        for r, g, b, a in colors:
            new_r = max(0, r-width)
            new_g = max(0, g-width)
            new_b = max(0, b-width)
            outlineColors.append((new_r, new_g, new_b))
        return outlineColors
    
    def plotTermProj(self, plt, pen, xDta, yDta, tickWidth, times=None):
        # If times are passed, add a color/time legend to the plot
        # and map each point to a color
        if times is not None:
            # Add a color bar / time legend to the plot
            startTime, endTime = times[0], times[-1]
            epoch = self.outerFrame.epoch
            plt.addColorLegend(epoch, startTime, endTime)

            # Map time values to RGB colors
            colorMap = LinearRGBGradient().getColorMap(startTime, endTime)
            colors = colorMap.map(times)
            brushes = list(map(pg.mkBrush, colors))

            # Make outlines darker
            outlineColors = self.getOutlineColors(colors)
            pens = list(map(pg.mkPen, outlineColors))

            plt.scatterPlot(xDta, yDta, size=5, pen=pens, brush=brushes)
        else: # Otherwise, plot a simple scatter plot
            brush = pg.mkBrush(pen.color())
            plt.scatterPlot(xDta, yDta, size=2, pen=pen, brush=brush)

    def plotMagLines(self, plt, pen, xField, yField, xDta, yDta, tickWidth):
        # Scale magnetic field data
        scaleFactor = self.getScalingFactor()
        xField = xField / scaleFactor
        yField = yField / scaleFactor

        # Adjust origin points for each vector if center lines is checked
        centerLines = self.ui.centerLinesBox.isChecked()
        if centerLines:
            width = xField / 2
            height = yField / 2

            xDta = xDta - width
            yDta = yDta - height

        xEnd = xDta + xField
        yEnd = yDta + yField

        # Plot vectors
        plt.plotMagLines(xDta, yDta, xEnd, yEnd, tickWidth, pen)

        # Add in scale bar item
        scaleBar = plt.addScaleBar(scaleFactor)

        return (centerLines, xField, yField)

    def getTextAngle(self, xDta, yDta):
        # Rotate text by 45 degrees if the average slope is > 0.5
        xDiff = max(xDta) - min(xDta)
        yDiff = max(yDta) - min(yDta)
        posDiff = np.array([xDiff, yDiff])
        posDiff = posDiff / np.linalg.norm(posDiff)
        xAxisVec = [1 * np.sign(xDiff), 0]

        # Calculate the angle between the x-axis and the vector (xMax - xMin, 
        # yMax - yMin)
        cos_alpha = np.dot(posDiff, xAxisVec)
        alpha = np.rad2deg(np.arccos(cos_alpha))

        if alpha < 18:
            angle = 45
        else:
            angle = 0

        return angle

    def getTextAnchor(self, posDta, angle, fieldDta=None):
        xDta, yDta = posDta
        # Anchor text to the left of point if the orbit is close to
        # the origin and on its left side
        if fieldDta is None:
            magDta = np.sqrt((xDta ** 2) + (yDta ** 2))
            anchor = (0, 0.5)
            minIndex = np.argmin(magDta)
            minDist = magDta[minIndex]
            minX = xDta[minIndex]
            originRadius = self.getOriginRadius(self.outerFrame.getPosVec())
            if (minDist < originRadius * 4 and minX < 0 and max) and angle == 0:
                anchor = (1, 0.5)
        else:
            # Try to find the side of the orbit line where most of the vectors
            # are pointing towards
            xField, yField = fieldDta
            xFAvg = np.mean(xField)
            yFAvg = np.mean(yField)
            fieldAvg = np.array([xFAvg, yFAvg])
            fieldAvg = fieldAvg / np.linalg.norm(fieldAvg)

            # Orbit line
            center, width = int(len(xDta) / 2), 5
            lb, rb = max(0, center-width), min(len(xDta), center+width)
            xDiff = xDta[lb] - xDta[rb]
            yDiff = yDta[lb] - yDta[rb]
            posDiff = np.array([xDiff, yDiff])
            posDiff = posDiff / np.linalg.norm(posDiff)

            # Difference vector
            diffVec = fieldAvg - posDiff
            refDiff = diffVec[1] if angle != 0 else diffVec[0]

            if refDiff > 0:
                anchor = (1, 0.5)
            else:
                anchor = (0, 0.5)

        return anchor

    def getTickWidth(self, xDta, yDta, ticks=True):
        # Estimate a reasonable tick width from position data
        distFrac = 0.02
        if ticks: # If there are markers on the plot
            # Min of (spacing between points / 2) and max distance * distFrac
            sampleWidth = min(10, len(xDta))
            xSample = np.diff(xDta[:sampleWidth])
            ySample = np.diff(yDta[:sampleWidth])
            allDist = np.sqrt((ySample ** 2) + (xSample ** 2))
            avgDist = np.mean(allDist)

            totDist = self.getMaxWidth(xDta, yDta)
            width = min(abs(avgDist / 2), totDist * distFrac)
            width = max(width, totDist * 0.001)
            return width
        else:
            totDist = self.getMaxWidth(xDta, yDta)
            return totDist * distFrac

    def addOriginItem(self, val):
        # Dynamically add or remove origin graphic from plot
        if self.currPlot is None:
            return

        if val:
            posDstrs = self.outerFrame.getPosVec()
            originRadius = self.getOriginRadius(posDstrs)
            self.currPlot.addOriginItem(originRadius)
        else:
            self.currPlot.removeOriginItem()

    def plotMagnetosphere(self, plt, t0, xDta, yDta):
        if not self.magTool:
            self.initMagTool()

        # Show status message during delay        
        self.outerFrame.ui.statusBar.showMessage('Calculating model field line coordinates...')
        self.outerFrame.ui.processEvents()

        # Get the axis numbers to plot and pass them to the magnetosphere tool
        # to calculate the field line coordinates
        aAxisNum, bAxisNum = self.getViewAxes()
        xcoords, ycoords, tiltAngle = self.magTool.getFieldLines(aAxisNum, bAxisNum)

        # Plot each trace and print out the dipole tilt angle
        pen = pg.mkPen('#d47f00')
        pen.setWidthF(1.1)
        for x, y in zip(xcoords, ycoords):
            plt.plot(x,y, pen=pen)
        self.outerFrame.ui.statusBar.showMessage('Dipole Tilt Angle: '+str(tiltAngle))
    
    def plotOrbitLine(self, plt, xDta, yDta, tickType, gaps, pen, width):
        # Use a black pen for orbit lines when also drawing field lines
        fieldLinePlot = (tickType == 'Field Lines' or tickType == 'Both')
        arrowPen = pg.mkPen('#000000') if fieldLinePlot else pen
        plt.plot(xDta, yDta, pen=arrowPen, connect=gaps)

        # Draw an arrow at end of orbit line
        if tickType == 'None':
            plt.plotArrowLine(xDta[-2:], yDta[-2:], width, pen)

    def calcOffset(self, angle, anchor, centerLines):
        ofst = 15 if centerLines else 5 # Large offset when overlapping w/ lines
        xAnchor, yAnchor = anchor

        # Opposite direction for labels oriented to left/bottom
        if xAnchor == 1:
            ofst = -ofst

        if angle != 0:
            offsets = (ofst, -ofst)
        else:
            offsets = (ofst, 0)

        return offsets

    def getTimeTickValues(times, epoch, gaps, td=None):
        # Initialize a datetime axis to generate ticks
        minTime, maxTime = times[0], times[-1]
        ta = DateAxis(epoch, orientation='bottom')
        ta.setRange(minTime, maxTime)

        if td: # Set user-specified custom tick spacing
            ta.setCstmTickSpacing(td)

            # Determine which time units need to be included in labels
            refLst = []
            hours = int(td.total_seconds() / (60 * 60))
            minutes = int((td.total_seconds() - (hours*60*60))/ 60)
            seconds = int(td.total_seconds() - (hours*60*60 + minutes*60))
            for val, itemStr in zip([hours, minutes, seconds],
                ['HH', 'MM', 'SS']):
                if val != 0:
                    refLst.append(itemStr)

            refStr = refLst[0] if len(refLst) > 0 else ''
            for i in range(1, len(refLst)):
                refStr = refStr + ':' + refLst[i]

            # Set the time axis label format to one that includes all the
            # necessary time units
            for axisMode in ta.timeModes[4:]:
                if refStr in axisMode:
                    ta.setLabelFormat(axisMode)
                    break

        # Get the default tick values/levels
        tickValues = ta.tickValues(minTime, maxTime, 500)

        # Extract enough ticks so that there are at least 3 time ticks
        minLevels = 3 if td is None else 1
        vals = []
        for i in range(0, min(minLevels, len(tickValues))):
            levelSpacing, levelVals = tickValues[i]
            if len(vals) <= 2:
                vals.extend(levelVals)

        # Ignore any time ticks outside of time range
        vals = [t for t in vals if t >= minTime and t <= maxTime]
        vals.sort()

        # Find time gap ranges and remove time ticks in this range
        gapTimes = []
        indices = np.arange(0, len(times))
        gapIndices = [indices[i] for i in range(0, len(indices)) if gaps[i] == 0]
        for i in gapIndices:
            if i > 0 and i+1 < len(times):
                gapTimes.append((times[i], times[i+1]))

        gapVals = []
        for t in vals:
            for gapStart, gapEnd in gapTimes:
                if t > gapStart and t < gapEnd:
                    gapVals.append(t)
        vals = [t for t in vals if t not in gapVals]

        return vals, ta

    def getTimeTickPositions(xFull, yFull, times, epoch, gaps=None, td=None):
        ticks, axis = OrbitPlotter.getTimeTickValues(times, epoch, gaps, td)
        if len(ticks) < 1:
            return None

        # Interpolate position data along the evenly spaced time tick values
        # and plot points at these coordinates
        csX = interpolate.CubicSpline(times, xFull)
        csY = interpolate.CubicSpline(times, yFull)

        xInterp = csX(ticks)
        yInterp = csY(ticks)

        return xInterp, yInterp, ticks, axis

    def plotTimeTickLabels(plt, posDta, tickVals, timeAxis, window, anchInfo=None):
        xInterp, yInterp = posDta
        anchor, angle, ofst = (0, 0.5), 0, (2, 0)
        if anchInfo is not None:
            anchor, angle, ofst = anchInfo

        # Generate labels for each time tick and add to plot
        labels = []
        for x, y, t in zip(xInterp, yInterp, tickVals):
            # Format timestamp
            fmt = timeAxis.get_fmt_str()
            date = ff_time.tick_to_date(t, timeAxis.epoch)
            timestamp = date.strftime(fmt)

            lbl = TimeTickLabel(timestamp, color='#000000', anchor=anchor, angle=angle)
            labels.append(lbl)

            plt.addItem(lbl)
            lbl.setPos(x,y)
            lbl.setOffset(ofst)
        return labels

    def plotTimesAlongOrbit(self, plt, posDta, magDta, timeDta, gaps, td=None):
        # Extract data and parameters
        epoch = self.outerFrame.epoch
        fieldLinesPlotted = self.ui.fieldLineBox.isChecked()
        centerLines = self.ui.centerLinesBox.isChecked() and fieldLinesPlotted
        times, fullTimes = timeDta
        xDta, yDta, xFull, yFull = posDta
        xMag, yMag, xMagFull, yMagFull = magDta

        # Plot starting and ending ticks
        brush = pg.mkBrush('#FFFFFF') # White fill, black outline
        pen = pg.mkPen('#000000')
        plt.scatterPlot([xFull[0]], [yFull[0]], symbol='s', pen=pen, size=8, brush=brush)
        plt.scatterPlot([xFull[-1]], [yFull[-1]], symbol='o', pen=pen, size=8, brush=brush)

        # Initialize a datetime axis and generate the tick values to plot
        res = OrbitPlotter.getTimeTickPositions(xFull, yFull, fullTimes, epoch, gaps, td)
        if res is None:
            return
        ## Unpack time tick positions, time ticks, and a datetime axis
        xInterp, yInterp, tickValues, timeAxis = res

        plt.scatterPlot(xInterp, yInterp, symbol='o', size=6, pen=pen, brush=brush)

        # Determine angle, anchor, and offset for the text labels
        angle = self.getTextAngle(xDta, yDta)
        fieldDta = (xMag, yMag) if fieldLinesPlotted else None
        anchor = self.getTextAnchor((xDta, yDta), angle, fieldDta)
        ofst = self.calcOffset(angle, anchor, centerLines)

        # Plot time tick labels using given tick values/positions and anchoring
        # information
        anchInfo = (anchor, angle, ofst)
        labels = OrbitPlotter.plotTimeTickLabels(plt, (xInterp, yInterp), 
            tickValues, timeAxis, self.outerFrame, anchInfo=anchInfo)

        # Add an additional label to indicate the timestamp format
        axisLabel = timeAxis.get_label()
        spacing = tickValues[1] - tickValues[0]
        fracSpacing = abs(spacing / 4)
        diff = abs(tickValues[0] - times[0])
        if diff < fracSpacing: # Append to first tick if very close to start time
            lbl = labels[0]
            label = lbl.getText()
            a, b = anchor
            axisLabel = '(' + axisLabel + ')'
            if a == 1: # Place on separate line
                anchor = (a, 0.25)
                diff = len(axisLabel) - len(label)
                startStr = ' ' * (abs(diff)*2)
                if diff < 0: # Axis label is longer
                    axisLabel = startStr + axisLabel
                else:
                    label = startStr + label
                label = label + '\n' + axisLabel
            else: # Add to end of string
                label = label + ' ' + axisLabel
            lbl.setPlainText(label)
        else:
            # Otherwise, add to start of orbit
            lbl = TimeTickLabel(axisLabel, color='#000000', anchor=anchor, angle=angle)
            plt.addItem(lbl)
            lbl.setOffset(ofst)
            lbl.setPos(xFull[0], yFull[0])

    def getTimeInterval(self):
        # Returns the custom time tick spacing set by the user
        if self.ui.autoBtn.isChecked():
            return None
        value = self.ui.timeBox.time()
        formatStr = '%H:%M:%S'
        minBoxTime = datetime.strptime(QtWidgets.QTimeEdit().minimumTime().toString(), formatStr)
        valTime = datetime.strptime(value.toString(), formatStr)
        value = valTime - minBoxTime
        return value

    def checkTimeTickInterval(self, td, t0, t1, tickType, projMode):
        # Don't check if not plotting time ticks
        if projMode or tickType not in ['Both', 'Time Ticks'] or td is None:
            return True

        # Check if time tick interval is less than total time / 150 or is
        # greater than half the the total time, printing an error message if so
        diff = abs(t1-t0)
        frac = diff / 150
        interval = abs(td.total_seconds())
        if interval < frac:
            msg = 'Error: Time tick interval is too small'
            self.outerFrame.ui.statusBar.showMessage(msg, 2500)
            return False
        elif interval > (diff / 2):
            msg = 'Error: Time tick interval too large'
            self.outerFrame.ui.statusBar.showMessage(msg, 2500)
            return False
        
        return True

    def getStartEndTimes(self):
        # Returns the overall region selected for the Trajectory Analysis tool
        startDt = self.outerFrame.ui.timeEdit.start.dateTime()
        endDt = self.outerFrame.ui.timeEdit.end.dateTime()
        return (startDt, endDt)

    def getEarthRadius(self):
        return self.outerFrame.getEarthRadius()

    def getMinMaxDt(self):
        return self.outerFrame.window.getMinAndMaxDateTime()

    def updatePlot(self):
        # Initialize plot parameters/scales if first plot
        if not self.scaleSet:
            self.initValues()

        # Get user-set parameters and
        projMode = self.ui.projBtn.isChecked()
        en = self.outerFrame.getEditNum()
        sI, eI = self.outerFrame.getIndices()
        vecDstrs = self.outerFrame.getFieldVec()
        posDstrs = self.outerFrame.getPosVec()
        radius = self.outerFrame.getRadius()
        times = self.outerFrame.getTimes(vecDstrs[0], en)
        times = times[sI:eI]
        fullTimes = times[:]
        t0, t1 = times[0], times[-1] # Save start/end times for time label
        td = self.getTimeInterval()

        if not self.checkTimeTickInterval(td, t0, t1, self.getTickType(), projMode):
            return

        # Get position data corresponding to plane chosen by user
        if projMode:
            xDta, yDta = self.getProjTermDta(posDstrs, vecDstrs, sI, eI, en)
            xField, yField = [], []
        else:
            xDta, yDta, xField, yField = self.getPosAndField(posDstrs, vecDstrs, sI, eI)

        # Scale if necessary
        xDta = xDta * radius
        yDta = yDta * radius
        xFull = xDta[:]
        yFull = yDta[:]
        xFieldFull = xField[:]
        yFieldFull = yField[:]

        # Create plot item and initialize pens
        pen = self.outerFrame.getPens()[0]
        orbitPen = pg.mkPen(pen.color())
        plt = OrbitPlotItem()

        # Plot optional magnetosphere model field lines first (lowest level)
        if self.ui.modelBox.isChecked():
            self.plotMagnetosphere(plt, t0, xDta, yDta)

        # Limit data/times by the user's sample rate if plotting markers
        tickType = self.getTickType() if not projMode else 'None'
        if tickType != 'None' or projMode:
            if projMode:
                rate = self.ui.projIntBox.value()
            else:
                rate = self.ui.intervalBox.value()
            xDta = self.sampleData(xDta, times, rate)
            yDta = self.sampleData(yDta, times, rate)
            xField = self.sampleData(xField, times, rate)
            yField = self.sampleData(yField, times, rate)
            times = self.sampleData(times, times, rate)

        # Mask out data points where an error flag is seen for any value
        mask = [True] * len(times)
        for arr in [xDta, yDta, xField, yField, times]:
            if len(arr) < 1:
                continue
            subMask = arr < self.outerFrame.window.errorFlag
            mask = mask & (subMask) # AND operation between masks

        xDta = xDta[mask]
        yDta = yDta[mask]
        xField = xField[mask] if len(xField) > 0 else xField
        yField = yField[mask] if len(yField) > 0 else yField
        times = times[mask]
        gaps = self.outerFrame.getSegments(sI, eI)

        # Get tick marker width
        tickWidth = self.getTickWidth(xDta, yDta, tickType != 'None')

        colorMode = self.ui.colorMapBox.isChecked()
        # Plot orbit and its arrow to indicate direction
        if not projMode:
            self.plotOrbitLine(plt, xFull, yFull, tickType, gaps, orbitPen, tickWidth)

        # Add additional markers onto plot
        magDta = None
        if projMode:
            colorTimes = times if colorMode else None
            self.plotTermProj(plt, pen, xDta, yDta, tickWidth, colorTimes)
        if tickType == 'Field Lines' or tickType == 'Both':
            magDta = self.plotMagLines(plt, pen, xField, yField, xDta, yDta, tickWidth)
            opt, xF, yF = magDta
            magDta = (opt, xF, yF, times)
        if tickType == 'Time Ticks' or tickType == 'Both':
            magDta = (xField, yField, xFieldFull, yFieldFull)
            timesDta = (times, fullTimes)
            posDta = (xDta, yDta, xFull, yFull)
            self.plotTimesAlongOrbit(plt, posDta, magDta, timesDta, gaps, td)

        # Origin item
        if self.ui.pltOriginBox.isChecked():
            originRadius = self.getOriginRadius(posDstrs)
            plt.addOriginItem(originRadius)

        # Set plot labels
        unitLbl = self.outerFrame.getRadiusUnits()
        unitLbl = ' (' + unitLbl + ')' if unitLbl != '' else ''
        ax_y = self.ui.planeBox1.currentText() if not projMode else 'Z'
        ax_x = self.ui.planeBox2.currentText() if not projMode else 'Y'
        xLbl = ax_x + unitLbl
        yLbl = ax_y + unitLbl
        plt.getAxis('bottom').setLabel(xLbl)
        plt.getAxis('left').setLabel(yLbl)
        plt.setTitle(self.ui.plotTitleBox.text())

        # Clear grid and add plot
        self.ui.glw.clear()
        self.ui.glw.addItem(plt, 0, 0, 1, 1)
        self.addTimeInfo(t0, t1)
        self.currPlot = plt
        self.lockAspect(self.ui.aspectBox.isChecked())

class TimeTickLabel(pg.TextItem):
    def __init__(self, text='', color=(200,200,200), html=None, anchor=(0,0),
                 border=None, fill=None, angle=0, rotateAxis=None, offset=(0,0)):
        self.offset = offset
        pg.TextItem.__init__(self, text, color, html, anchor, border, fill,
            angle, rotateAxis)

    def setOffset(self, ofst):
        self.offset = ofst
        self.updateTextPos()

    def updateTextPos(self):
        # update text position to obey anchor
        r = self.textItem.boundingRect()
        tl = self.textItem.mapToParent(r.topLeft())
        br = self.textItem.mapToParent(r.bottomRight())
        ofstX, ofstY = self.offset
        offset = (br - tl) * self.anchor
        offset = QtCore.QPointF(offset.x()-ofstX, offset.y()-ofstY)
        self.textItem.setPos(-offset)

    def getText(self):
        return self.textItem.toPlainText()

class MagnetosphereToolUI(BaseLayout):
    def setupUI(self, frame, outerFrame):
        frame.resize(400, 150)
        frame.setWindowTitle('Magnetosphere Model Settings')
        layout = QtWidgets.QGridLayout(frame)

        self.closeBtn = QtWidgets.QPushButton('Close')
        self.closeBtn.setSizePolicy(self.getSizePolicy('Max','Max'))
        parmFrm = self.setupParmFrame(frame, outerFrame)
        settingsFrm = self.setupSettingsFrame(frame, outerFrame)
        layout.addWidget(parmFrm, 0, 0, 1, 2)
        layout.addWidget(settingsFrm, 1, 0, 1, 2)
        layout.addWidget(self.closeBtn, 2, 1, 1, 1, QtCore.Qt.AlignRight)

        self.fixedTime = False
        self.refTimeBox.dateTimeChanged.connect(self.setFixedTime)

    def setupParmFrame(self, frame, outerFrame):
        parmFrame = QtWidgets.QGroupBox('Tsyganenko (T96) Model Parameters')
        layout = QtWidgets.QGridLayout(parmFrame)

        boxLbls = ['Solar Wind Ram Pressure:', 'DST Index:', 'IMF By:', 'IMF Bz:']
        boxTips = ['', 'Disturbance Storm-Time Index', 
            'Interplanetary Magnetic Field By Component',
            'Interplanetary Magnetic Field Bz Component']
        boxUnits = ['nPa', 'nT', 'nT', 'nT']

        # Create the spinboxes
        self.swrPressureBox = QtWidgets.QDoubleSpinBox()
        self.dstIndexBox = QtWidgets.QDoubleSpinBox()
        self.imfByBox = QtWidgets.QDoubleSpinBox()
        self.imfBzBox = QtWidgets.QDoubleSpinBox()
        boxes = [self.swrPressureBox, self.dstIndexBox, self.imfByBox, self.imfBzBox]

        # Set the upper/lower bounds for the model parameters
        boxRanges = [(-100, 20), (-30, 30), (-30, 30)]
        for box, (minVal, maxVal) in zip(boxes[1:], boxRanges):
            box.setMinimum(minVal)
            box.setMaximum(maxVal)

        # Add each parameter to its own row and set the labels/tooltips and units
        row = 0
        for box, lbl, units, tt in zip(boxes, boxLbls, boxUnits, boxTips):
            box.setSuffix(' '+units)
            self.addPair(layout, lbl, box, row, 0, 1, 1, tooltip=tt)
            row += 1

        return parmFrame

    def setupSettingsFrame(self, frame, outerFrame):
        # Time edit used for reference time
        settingsFrm = QtWidgets.QGroupBox('Other Settings')
        layout = QtWidgets.QGridLayout(settingsFrm)
        te = TimeEdit()
        self.refTimeBox = te.start
        te.setupMinMax(outerFrame.getMinMaxDt())

        # Coordinate system box
        self.coordBox = QtWidgets.QComboBox()
        self.coordBox.addItems(['GSM', 'GSE', 'SM'])
        self.coordBox.setCurrentIndex(0)

        # Boundary radius / rLim parameter box
        self.rlimBox = QtWidgets.QDoubleSpinBox()
        self.rlimBox.setMinimum(1)
        self.rlimBox.setMaximum(50)
        self.rlimBox.setValue(10)
        self.rlimBox.setSuffix(' RE')

        timeTip = 'Time used to calculate the dipole tilt angle'
        coordTip = 'Coordinate system of the model field lines'
        radiusTip = 'Outer boundary that field lines stop at relative to the origin, RE = 6731.2 km'
        boxes = [self.refTimeBox, self.coordBox, self.rlimBox]
        boxLbls = ['Reference Time: ', 'Coordinate System:', 'Boundary Radius:']
        boxTips = [timeTip, coordTip, radiusTip]
        row = 0
        for box, lbl, tt in zip(boxes, boxLbls, boxTips):
            self.addPair(layout, lbl, box, row, 1, 1, 1, tooltip=tt)
            row += 1

        return settingsFrm

    def setFixedTime(self):
        self.fixedTime = True

class MagnetosphereTool(QtWidgets.QFrame, MagnetosphereToolUI):
    def __init__(self, outerFrame, parent=None):
        super().__init__(parent)
        self.ui = MagnetosphereToolUI()
        self.outerFrame = outerFrame
        self.ui.setupUI(self, outerFrame)

        # Constants
        self.earthRadius = self.outerFrame.getEarthRadius()
        self.unitCoords = self.getUnitCoords()

        # Determine number of threads available
        self.numThreads = 1
        try:
            numProcs = multiprocessing.cpu_count()
        except:
            numProcs = 1
        if numProcs >= 8:
            self.numThreads = 8
        elif numProcs >= 4:
            self.numThreads = 4
        elif numProcs >= 2:
            self.numThreads = 2
        else:
            self.numThreads = 1

        # Set default box values
        self.ui.swrPressureBox.setValue(0.5)
        self.ui.imfByBox.setValue(6)
        self.ui.imfBzBox.setValue(-5)
        self.ui.dstIndexBox.setValue(-30)

        # Connect close button to closeWindow event
        self.ui.closeBtn.clicked.connect(self.close)

    def getState(self):
        modelParams = self.getModelParameters()
        otherParams = self.getOtherParameters()
        return (modelParams, otherParams)

    def loadState(self, state):
        modelParams, otherParams = state

        # Load model parameters
        swrp, dstIndex, imfBy, imfBz = modelParams
        self.ui.swrPressureBox.setValue(swrp)
        self.ui.dstIndexBox.setValue(dstIndex)
        self.ui.imfByBox.setValue(imfBy)
        self.ui.imfBzBox.setValue(imfBz)

        # Load general parameters
        dt, coordSys, rLim = otherParams
        self.ui.refTimeBox.setDateTime(dt)
        self.ui.rlimBox.setValue(rLim)
        self.ui.coordBox.setCurrentText(coordSys)

    def updtTime(self):
        # Update time if it hasn't been set or if the set time is out of
        # the selected time range
        frameTime, frameEnd = self.outerFrame.getStartEndTimes()
        currTime = self.ui.refTimeBox.dateTime()
        outOfRange = (self.ui.fixedTime and (frameTime > currTime) or (frameEnd < currTime))
        if not self.ui.fixedTime or outOfRange:
            self.ui.refTimeBox.blockSignals(True)
            self.ui.refTimeBox.setDateTime(frameTime)
            self.ui.refTimeBox.blockSignals(False)
            self.ui.fixedTime = False

    def getUnitCoords(self):
        # Points around the unit circle and an additional set of points
        # beyond each side of the dipole (at 180 and 0 degrees)
        angles = list(np.arange(0, 360, 15))
        for angle in [0, 180, 195, 345]:
            angles.remove(angle)
        baseCoords = [(np.cos(np.radians(angle)), np.sin(np.radians(angle))) for angle in angles]
        for scale in [1.5, 2.5, 3]:
            for angle in [0, 180]:
                coord = (scale*np.cos(np.radians(angle)), scale*np.sin(np.radians(angle)))
                baseCoords.append(coord)

        return baseCoords

    def getUniversalTime(self, dt):
        # Seconds since 1970
        baseTime = datetime(1970, 1, 1)
        diff = dt - baseTime
        return diff.total_seconds()

    def getModelParameters(self):
        swrp = self.ui.swrPressureBox.value()
        dstIndex = self.ui.dstIndexBox.value()
        imfBy = self.ui.imfByBox.value()
        imfBz = self.ui.imfBzBox.value()
        return swrp, dstIndex, imfBy, imfBz

    def getOtherParameters(self):
        dt = self.ui.refTimeBox.dateTime().toPyDateTime()
        coordSys = self.ui.coordBox.currentText()
        rLim = self.ui.rlimBox.value()
        return dt, coordSys, rLim

    def getFieldLines(self, axisNum1, axisNum2):
        # Update time if selection is out of plot time range
        self.updtTime()

        # Extract parameters and other plot settings
        self.earthRadius = self.outerFrame.getEarthRadius()
        dt, coordSys, rLim = self.getOtherParameters()
        swrp, dstIndex, imfBy, imfBz = self.getModelParameters()

        r0 = 1 # All lines stop on surface of earth (1 RE)

        # Create a parmod object to pass to the geopack trace function
        parmod = np.zeros(10)
        parmod[0:4] = [swrp, dstIndex, imfBy, imfBz]

        # Call recalc from geopack to initialize its trace function variables and
        # calculate the dipole tilt angle
        univTime = self.getUniversalTime(dt)
        dipoleTilt = geopack.recalc(univTime)

        # Get the coordinates to plot for the given pair of axes,
        # rotated by dipole tilt angle to make sure the dipole is
        # aligned w/ the 'unit circle' the points were derived from
        plotCoords = self.getCoords(axisNum1, axisNum2, dipoleTilt)

        # Calculate the field line values from the model and store the results,
        # using multiple processes if appropriate
        xCoords, yCoords, zCoords = [], [], []
        coordLists = [xCoords, yCoords, zCoords]
        if self.numThreads == 1 or len(plotCoords) < self.numThreads:
            for vecDir in [1, -1]:
                for p in plotCoords:
                    x, y, z = MagnetosphereTool.getFieldLine(p, parmod, vecDir,
                    rLim, r0)
                    for lst, axisLine in zip(coordLists, [x,y,z]):
                        lst.append(np.array(axisLine))
        else:
            coordLists = self.multiProcFieldLines(plotCoords, parmod, r0, rLim, univTime)

        # Scale the coordinates if the plot units are in kilometers instead of RE
        self.scaleCoords(coordLists[0], coordLists[1], coordLists[2])
        plotCoords = [(x*self.earthRadius, y*self.earthRadius, z*self.earthRadius) for x,y,z in plotCoords]

        # Change coordinates if output is GSE
        if coordSys == 'GSE':
            coordLists = self.coordsToGSE(coordLists[0], coordLists[1], coordLists[2])
        elif coordSys == 'SM': # Convert to SM coordinates
            coordLists = self.coordsToSM(coordLists[0], coordLists[1], coordLists[2])

        # Dipole tilt angle returned should be in degrees
        dipoleTilt = np.degrees(dipoleTilt)
        return coordLists[axisNum2], coordLists[axisNum1], dipoleTilt

    def multiProcFieldLines(self, plotCoords, parmod, r0, rLim, univTime):
        # Multiprocessed calculation of the field lines
        # Get number of lines to calculate per process and split coords into groups
        grpSize = int(len(plotCoords) / self.numThreads)
        grps = []
        lastIndex = 0
        for startIndex in range(0, len(plotCoords), grpSize):
            parmList = (parmod[:], rLim, r0, univTime)
            grps.append((plotCoords[startIndex:startIndex+grpSize], parmList))
            lastIndex = startIndex + grpSize

        if lastIndex < len(plotCoords) - 1:
            grps.append((plotCoords[lastIndex:len(plotCoords)], parmList))

        # Create a process pool and run the geopack trace function for each
        # of them with a subset of the plot coordinates
        pool = Pool(processes=self.numThreads)
        res = pool.map(MagnetosphereTool.calcFieldLinesForCoords, grps)

        # Extract coordinates from pool result
        xCoords, yCoords, zCoords = [], [], []
        for procRes in res:
            for lineList in procRes:
                lineX, lineY, lineZ = lineList
                xCoords.append(lineX)
                yCoords.append(lineY)
                zCoords.append(lineZ)

        return [xCoords, yCoords, zCoords]

    def scaleCoords(self, xCoords, yCoords, zCoords):
        # Scale coordinates by the earth radius if units are kilometers
        for xLine, yLine, zLine in zip(xCoords, yCoords, zCoords):
            xLine *= self.earthRadius
            yLine *= self.earthRadius
            zLine *= self.earthRadius

    def coordsToGSE(self, xCoords, yCoords, zCoords):
        # Convert coordinates from GSM to GSE
        return self.convertCoords(xCoords, yCoords, zCoords, 'GSE')

    def coordsToSM(self, xCoords, yCoords, zCoords):
        return self.convertCoords(xCoords, yCoords, zCoords, 'SM')

    def convertCoords(self, xCoords, yCoords, zCoords, coordSys):
        if coordSys == 'GSE': # Map from GSM to GSE
            mapFunc = lambda v : geopack.gsmgse(v[0], v[1], v[2], 100)
        else: # Map from GSM to SM
            mapFunc = lambda v : geopack.smgsm(v[0], v[1], v[2], -100)

        # Convert coordinates from GSM to either GSM or SM
        xNew, yNew, zNew = [], [], []
        for lineX, lineY, lineZ in zip(xCoords, yCoords, zCoords):
            newX, newY, newZ = [], [], []
            for x, y, z in zip(lineX, lineY, lineZ):
                xconv, yconv, zconv = mapFunc((x, y, z))
                newX.append(xconv)
                newY.append(yconv)
                newZ.append(zconv)
            xNew.append(newX)
            yNew.append(newY)
            zNew.append(newZ)
        return [xNew, yNew, zNew]

    def getCoords(self, axis1, axis2, tiltAngle, rScale=3.5):
        otherAxis = (set([0,1,2]) - set([axis1, axis2])).pop()
        baseCoords = self.unitCoords[:]
        if otherAxis == 2: # If plotting the XY plane, do not use scaled unit points
            baseCoords = self.unitCoords[:-6]
            baseCoords.append((1,0))
            baseCoords.append((-1,0))

        # Use points around the unit circle to set the coordinates for the
        # given axes, and set the third axis (pointing away from the plot)
        # to zero
        coords = []
        fillVal = 0
        for (x, y) in baseCoords:
            t = (x, y, fillVal)
            if otherAxis == 0:
                t = (fillVal, x, y)
            elif otherAxis == 1:
                t = (x, fillVal, y)

            coords.append(t)

        # Scale the points by 3.5 RE for now
        coords = [(x*rScale, y*rScale, z*rScale) for x,y,z in coords]

        # Rotate each coordinate by the tilt angle about the y axis
        rotMat = [[np.cos(tiltAngle), 0, np.sin(tiltAngle)],
                  [0, 1, 0],
                  [-np.sin(tiltAngle), 0, np.cos(tiltAngle)]]
        rotCoords = []
        for x, y, z in coords:
            x, y, z = np.matmul(rotMat, [x,y,z])
            p0 = (x, y, z)
            rotCoords.append(p0)

        return rotCoords

    def calcFieldLinesForCoords(args):
        # Unpack arguments for the multiprocessed version of the calculations
        coordList, parmList = args
        parmod, rLim, r0, univTime = parmList
        # Get line at each coordinate in both directions
        geopack.recalc(univTime)
        lines = []
        for vecDir in [-1, 1]:
            for p0 in coordList:
                line = MagnetosphereTool.getFieldLine(p0, parmod, vecDir, rLim, r0)
                lines.append(line)

        return lines

    def getFieldLine(p0, parmod, vecDir, rLim, r0):
        x0, y0, z0 = p0
        res = geopack.trace(x0, y0, z0, vecDir, rLim, r0, parmod, 't96', 
            'igrf', maxloop=50)
        x, y, z, xx, yy, zz = res
        return xx, yy, zz

# Default color gradient that starts w/ blue and ends in red
class LinearRGBGradient(QtGui.QLinearGradient):
    def __init__(self, *args, **kwargs):
        QtGui.QLinearGradient.__init__(self, *args, **kwargs)

        rgbBlue = (25, 0, 245)
        rgbBlueGreen = (0, 245, 245)
        rgbGreen = (127, 245, 0)
        rgbYellow = (245, 245, 0)
        rgbRed = (245, 0, 25)

        self.colorPos = [0, 1/3, 0.5, 2/3, 1]
        self.colors = [rgbBlue, rgbBlueGreen, rgbGreen, rgbYellow, rgbRed]
        for pos, rgb in zip(self.colorPos, self.colors):
            self.setColorAt(pos, pg.mkColor(rgb))

    def getColorMap(self, minVal, maxVal):
        minVal, maxVal = min(minVal, maxVal), max(minVal, maxVal)
        diff = maxVal - minVal

        values = []
        for pos in self.colorPos:
            val = minVal + diff*pos
            if pos == 1:
                val = maxVal
            values.append(val)

        return pg.ColorMap(values, self.colors)

class TimeColorBar(GradLegend, GraphicsWidgetAnchor):
    def __init__(self, epoch, parent=None, edge='right', *args, **kwargs):
        # Initialize state and legend elements
        self.valueRange = (0, 1)
        self.colorBar = ColorBar(QtGui.QLinearGradient())
        self.axisEdge = edge # Side of color bar that time axis is displayed on
        self.axisItem = DateAxis(epoch, orientation=edge)
        self.label = None # Axis time label

        # Set default contents spacing/margins
        pg.GraphicsLayout.__init__(self, parent)
        GraphicsWidgetAnchor.__init__(self)
        self.layout.setVerticalSpacing(0)
        self.layout.setHorizontalSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum))

        cbCol = 0 if edge == 'right' else 1
        axCol = 1 if edge == 'right' else 0
        self.addItem(self.colorBar, 0, cbCol, 1, 1)
        self.addItem(self.axisItem, 0, axCol, 1, 1)

        # Set up DateTime Axis
        self.axisItem.setStyle(tickLength=-10)
        self.setRange(LinearRGBGradient(), (0,1))

        # Set up time label, anchoring to the appropriate edge
        self.label = pg.LabelItem()
        self.label.opts['justify'] = 'left' if edge == 'right' else 'right'
        self.addItem(self.label, 1, 0, 1, 2)
        edgeOfst = 5 if edge == 'left' else -5
        self.label.anchor((1,1),(1,1),(edgeOfst, 0))

    def getOffsetSuggestions(self):
        if self.axisEdge == 'left':
            ofsts = (-15, 15)
        else:
            ofsts = (-12, 15)
        return ofsts
    
    def setRange(self, gradient, valRange):
        GradLegend.setRange(self, gradient, valRange)
        self.updateLabel()

    def updateLabel(self):
        # Set time axis label when value range is updated
        if self.label:
            lbl = self.axisItem.getDefaultLabel()
            self.label.setText(lbl)

class OrbitPlotItem(MagPyPlotItem):
    def __init__(self):
        self.originItem = None
        self.scaleBar = None
        self.colorBar = None

        MagPyPlotItem.__init__(self)

        for ax in ['bottom', 'left','top', 'right']:
            self.getAxis(ax).setStyle(tickLength=-10)
        self.hideButtons()

    def addOriginItem(self, radius=1, origin=(0,0), pen=None):
        if pen is None:
            pen = pg.mkPen((0,0,0))
        self.removeOriginItem()
        self.originItem = OriginGraphic(radius, origin, pen=pen)
        self.addItem(self.originItem)

    def removeOriginItem(self):
        if self.originItem:
            self.removeItem(self.originItem)
            self.originItem = None

    def addScaleBar(self, scaleAmt):
        vb = self.getViewBox()
        self.scaleBar = FieldScaleBar(vb, scaleAmt)
        return self.scaleBar

    def hideScaleBar(self, val=True):
        if self.scaleBar:
            if val:
                self.getViewBox().removeItem(self.scaleBar)
            else:
                self.scaleBar.setParentItem(self.getViewBox())

    def plotArrowLine(self, x, y, width, pen):
        line = ArrowLine(x,y,width,pen=pen)
        self.addItem(line)

    def plotMagLines(self, x, y, xF, yF, width, pen):
        brush = pg.mkBrush(pen.color())
        clstr = ArrowCluster(x, y, xF, yF, width, pen=pen, brush=brush)
        self.addItem(clstr)

    def addColorLegend(self, epoch, minTime, maxTime):
        # Initialize color bar legend and set the viewbox as its parent
        self.colorBar = TimeColorBar(epoch, edge='left')
        self.colorBar.setParentItem(self.getViewBox())
        self.colorBar.setRange(self.colorBar.getGradient(), (minTime, maxTime))

        # Anchor to top-right corner and set legend width/height
        self.colorBar.anchor((1,0), (1,0), offset=(-15, 15))
        self.colorBar.setFixedHeight(190)
        self.colorBar.setBarWidth(18)

        # Set top and right axes to be visible
        for ax in ['right', 'top']:
            self.showAxis(ax, True)
            self.getAxis(ax).setStyle(showValues=False)

    def removeColorLegend(self):
        if self.colorBar:
            self.getViewBox().removeItem(self.colorBar)
            self.colorBar = None

class ArrowLine(pg.PlotCurveItem):
    def __init__(self, x, y, width, *arg, **karg):
        self.arrowPath = None
        self.width = width
        pg.PlotCurveItem.__init__(self, x=x, y=y, *arg, **karg)

    def paint(self, p, opt, widget):
        pg.PlotCurveItem.paint(self, p, opt, widget)

        x, y = self.xData[1], self.yData[1]
        cp = pg.mkPen(self.opts['pen'])

        if self.arrowPath is None:
            self.arrowPath = self.buildArrow()

        p.fillPath(self.arrowPath, cp.color())

    def degreeToRad(self, degree):
        return np.deg2rad(degree)

    def getRotMat(self, angle):
        return [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]

    def norm(self, vec):
        return np.sqrt(np.dot(vec, vec))

    def buildArrow(self):
        x0, y0 = self.xData[-2], self.yData[-2]
        x1, y1 = self.xData[-1], self.yData[-1]
        endPt = np.array([x1, y1])

        width = self.width
        halfAngle = 18

        # Rotate difference vector by angle and scale it to have magnitude = width
        otherRefPt = np.array([x0-x1, y0-y1])
        mat1 = self.getRotMat(self.degreeToRad(halfAngle))
        vec1 = np.matmul(mat1, otherRefPt)
        norm1 = self.norm(vec1)
        vec1 = (vec1 / norm1) * width

        # Rotate the new vector to form other edge of arrow
        mat2 = self.getRotMat(self.degreeToRad(360-halfAngle*2))
        vec2 = np.matmul(mat2, vec1)

        # Add the difference vectors to reference point
        i1, j1 = endPt + vec1
        i2, j2 = endPt + vec2
        i3, j3 = x1, y1

        # Form a closed path between the two endpoints and the tip of the
        # vector's line
        path = QtGui.QPainterPath()
        path.moveTo(i1,j1)
        path.lineTo(QtCore.QPointF(i3, j3))
        path.lineTo(QtCore.QPointF(i2, j2))
        path.closeSubpath()

        return path

    def getArrowPath(self):
        if self.arrowPath is None:
            self.arrowPath = self.buildArrow()
        return self.arrowPath

class ArrowCluster(pg.PlotCurveItem):
    def __init__(self, xDta, yDta, xField, yField, width, *args, **kargs):
        self.xDta = xDta
        self.yDta = yDta
        self.xField = xField
        self.yField = yField
        self.width = width
        pg.PlotCurveItem.__init__(self, xDta, yDta, *args, **kargs)
        self.xData = np.concatenate([xDta, xField])
        self.yData = np.concatenate([yDta, yField])

    def getPath(self):
        if self.path is None:
            self.path, self.arrowPaths = self.generatePath()
        return self.path

    def generatePath(self):
        path = QtGui.QPainterPath()
        arrowPaths = QtGui.QPainterPath()
        for p in [path, arrowPaths]:
            p.setFillRule(QtCore.Qt.WindingFill)

        for x, y, xf, yf in zip(self.xDta, self.yDta, self.xField, self.yField):
            path.moveTo(x,y)
            path.lineTo(xf, yf)
            arrowLine = ArrowLine([x, xf], [y, yf], self.width)
            arrowPath = arrowLine.getArrowPath()
            arrowPaths.addPath(arrowPath)
        return path, arrowPaths

    def paint(self, p, opt, widget):
        pg.PlotCurveItem.paint(self, p, opt, widget)
        p.fillPath(self.arrowPaths, self.opts['brush'])

class OriginGraphic(pg.PlotCurveItem):
    # Draws a circle w/ the given radius at given point to represent origin in
    # coordinate system
    def __init__(self, radius=1, origin=(0, 0), *args, **kargs):
        self.radius = radius
        self.originX, self.originY = origin
        self.arrowPath = None
        xVals = [self.originX-radius, self.originX+radius]
        yVals = [self.originY-radius, self.originY+radius]
        pg.PlotCurveItem.__init__(self, x=xVals, y=yVals, *args, **kargs)
    
    def getPath(self):
        if self.path is None:
            x, y = self.getData()
            self.path = QtGui.QPainterPath()
            self.fillPath = None
            self._mouseShape = None
            if x is None or len(x) == 0 or y is None or len(y) == 0:
                return self.path
            center = QtCore.QPointF(self.originX, self.originY)
            self.path.addEllipse(center, self.radius, self.radius)
            self.path.moveTo(center)

        return self.path

    def paint(self, p, opt, widget):
        if self.xData is None or len(self.xData) == 0:
            return
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        path = self.getPath()
        cp = pg.mkPen(self.opts['pen'])
        p.setPen(cp)
        p.drawPath(path)

        cp.setWidth(1)
        cp.setCapStyle(QtCore.Qt.RoundCap)
        p.setPen(cp)
        center = QtCore.QPointF(self.originX, self.originY)
        p.drawPoint(center)

    def setDownsampling(self, ds, auto, method):
        pass

    def setClipToView(self, clip):
        pass

class FieldScaleBar(pg.ScaleBar):
    def __init__(self, vb, size, width=1, brush=None, pen=None, suffix='m', offset=None):
        # Set up scale bins
        self.scaleRatio = 1 / size
        self.invScale = size
        self.baseVals = [1,2,5]
        self.midPts = [2, 4.5, 8]
        self.baseLevel = 1 # Fraction to multiply baseVal by
        self.currVal = self.baseVals[0]*self.baseLevel
        self.size = self.currVal * self.scaleRatio
        self.valIndex = 0 # Which base val is being used
        self.pltVb = vb

        # Set default pen/brush info
        pen = pg.mkPen('#000000')
        brush = pg.mkBrush(pen.color())
        offset = (-10, -10)
        pg.ScaleBar.__init__(self, size, width, brush, pen, suffix, offset)

        # Replace text item
        self.text.setParentItem(None)
        self.text = pg.TextItem('1 nT', anchor=(0.5, 1), color=pen.color())
        self.text.setParentItem(self)

        # Set plot item as scale bar's parent
        self.setParentItem(vb)

    def adjustScaleBase(self, maxWidth):
        base = maxWidth
        self.baseLevel = 10 ** (len(str(int(base))) - 1)
        i = 0
        while i < len(self.baseVals) and self.midPts[i] * self.baseLevel < maxWidth:
            i += 1

        if i == 3:
            self.baseLevel *= 10
            i = 0

        self.valIndex = i
        self.currVal = self.baseVals[self.valIndex] * self.baseLevel
        self.size = self.currVal * self.scaleRatio

    def parentChanged(self):
        view = self.parentItem()
        if view is None:
            return
        view.sigRangeChanged.connect(self._updateBar)
        self._updateBar()

    def getWidth(self):
        # Returns 1/9th of Y range and then scales it by the nT/(pos units) ratio
        xRng, yRng = self.pltVb.viewRange()
        a, b = yRng
        diff = abs(b-a)
        return (diff / 9)*(self.invScale)

    def _updateBar(self):
        QtCore.QTimer.singleShot(50, self.updateBar)
        self.adjustScaleBase(self.getWidth()) # Adjust scale bar when view changes
        self.text.setPlainText(str(self.currVal)+' nT')
