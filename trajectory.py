from PyQt5 import QtGui, QtCore, QtWidgets
from MagPy4UI import TimeEdit
from PyQt5.QtWidgets import QSizePolicy
from layoutTools import BaseLayout
from pyqtgraphExtensions import LinkedAxis, DateAxis, MagPyPlotItem
import pyqtgraph as pg
import numpy as np
from mth import Mth
from datetime import timedelta
import sys
sys.path.insert(0, 'geopack')
from geopack import geopack
from datetime import datetime, timedelta
import multiprocessing
from multiprocessing import Pool
import time

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
        # frame = QtWidgets.QGroupBox('Vector and Units Settings')
        frame = QtWidgets.QFrame()
        frame.setSizePolicy(self.getSizePolicy('Min', 'Max'))
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(2, 2, 5, 2)

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
        self.radiusBox.setMaximum(1e10)
        self.radiusBox.setMinimum(0.0001)
        self.radiusBox.setValue(1)
        self.radiusBox.setDecimals(4)
        self.addPair(layout, ' Radius: ', self.radiusBox, 0, 4, 1, 1)

        # Radius units
        self.radUnitBox = QtWidgets.QLineEdit()
        self.radUnitBox.setMaxLength(15)
        defaultUnit = window.UNITDICT[posVecs[0][-1]]
        if 'R'.lower() in defaultUnit[0].lower():
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

class TrajectoryAnalysis(QtGui.QFrame, TrajectoryUI):
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

    def validState(self):
        # Checks if at least one field vec and pos vec could be identified
        return len(self.fieldVecs) > 0 and len(self.posVecs) > 0

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
        layout.setContentsMargins(0, 5, 0, 0)
        settingsFrame = self.setupSettingsFrame(Frame, outerFrame)
        self.glw = self.getGraphicsGrid()
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

    def linkAxes(self, val=None):
        if val is None:
            val = self.ui.linkBox.isChecked()

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
            la = LinkedAxis('left')
            ba = LinkedAxis('bottom')
            plt = MagPyPlotItem(axisItems={'bottom':ba, 'left':la})
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
        frame.setSizePolicy(self.getSizePolicy('Max', 'Max'))

        # Plot type frame
        pltFrm = QtWidgets.QGroupBox('Plot Type:')
        pltLt = QtWidgets.QGridLayout(pltFrm)

        # Plane of view
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
        separator = self.getWidgetSeparator()
        layout.addWidget(separator, 1, 0, 1, 5)

        # Add tick settings, plot settings, and magnetosphere model layouts
        tickLt = self.setupTickLt(outerFrame, Frame)
        pltLt = self.setupPlotSettingsLt(outerFrame, Frame)
        modelFrm = self.setupModelLt(outerFrame, Frame)
        layout.addLayout(tickLt, 2, 0, 1, 5)
        layout.addWidget(modelFrm, 4, 0, 1, 5)
        layout.addLayout(pltLt, 6, 0, 1, 5)

        # Add in cosmetic separators
        for row in [3,5,7]:
            self.addSeparator(layout, row, 5)

        # Update button
        self.updtBtn = QtWidgets.QPushButton('Update')
        layout.addWidget(self.updtBtn, 8, 4, 1, 1)
        return frame

    def addSeparator(self, layout, row, span):
        separator = self.getWidgetSeparator()
        layout.addWidget(separator, row, 0, 1, span)

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
        spacer.setMinimumHeight(20)
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

    def setupTickLt(self, outerFrame, Frame):
        layout = QtWidgets.QGridLayout()

        # Plot tick types
        self.btnBox = QtWidgets.QGroupBox('Marker Type:')
        self.markerBtns = []
        btnLt = QtWidgets.QHBoxLayout(self.btnBox)
        btnLbls = ['Field Lines', 'Time Ticks', 'None']
        for lbl in btnLbls:
            btn = QtWidgets.QRadioButton(lbl)
            self.markerBtns.append(btn)
            btnLt.addWidget(btn)
        self.markerBtns[0].setChecked(True)
        layout.addWidget(self.btnBox, 0, 0, 1, 3)

        # Tick interval box
        intLt = QtWidgets.QGridLayout()
        self.intervalBox = QtWidgets.QSpinBox()
        self.intervalBox.setMinimum(1)
        self.intervalBox.setMaximum(100000)
        self.intervalBox.setSuffix(' pts')
        self.intervalBox.setSingleStep(5)
        tt = 'Number of points between time ticks/field lines'
        lbl = self.addPair(intLt, 'Tick Spacing:  ', self.intervalBox, 0, 0, 1, 1, tt)
        self.intLbl = lbl
        spacer = self.getSpacer(5)
        intLt.addItem(spacer, 0, 2, 1, 1)

        # Tick time representation label
        self.timeLbl = QtWidgets.QLabel()
        self.timeLbl.setToolTip('HH:MM:SS.SSS')
        self.intervalBox.valueChanged.connect(self.adjustTimeLabel)
        intLt.addWidget(self.timeLbl, 0, 3, 1, 1)
        layout.addLayout(intLt, 1, 0, 1, 3)

        # Field line scale
        scaleLt = QtWidgets.QGridLayout()
        self.magLineScaleBox = QtWidgets.QDoubleSpinBox()
        self.magLineScaleBox.setMinimum(1e-5)
        self.magLineScaleBox.setMaximum(1e10)
        self.magLineScaleBox.setDecimals(5)
        self.magLineScaleBox.setValue(0.25)
        self.magLineScaleBox.setMaximumWidth(200)
        tt = 'Ratio between field values and position data units'
        lbl = self.addPair(scaleLt, 'Field Line Scale:  ', self.magLineScaleBox, 2, 0, 1, 2, tt)
        self.magLineLbl = lbl
        layout.addLayout(scaleLt, 2, 0, 1, 3)

        # Center field lines check
        tt = 'Toggle to center field vectors around the orbit line'
        self.centerLinesBox = QtWidgets.QCheckBox('Center Field Vectors')
        self.centerLinesBox.setChecked(True)
        self.centerLinesBox.setToolTip(tt)
        layout.addWidget(self.centerLinesBox, 3, 0, 1, 1)

        return layout

    def setupModelLt(self, outerFrame, Frame):
        self.modelBox = QtWidgets.QGroupBox(' Plot Model of Earth\'s Magnetosphere')
        layout = QtWidgets.QHBoxLayout(self.modelBox)
        self.modelBox.setCheckable(True)
        self.modelBox.setChecked(False)
        self.modelBox.toggled.connect(self.modelChecked)
        self.modelBtn = QtWidgets.QPushButton('Set Model Parameters')
        self.modelBtn.setFixedWidth(235)

        layout.addWidget(self.modelBtn, QtCore.Qt.AlignCenter)
        return self.modelBox

    def adjustTimeLabel(self, val):
        vecDstrs = self.outerFrame.getFieldVec()
        dstr = vecDstrs[0]
        en = self.outerFrame.getEditNum()
        times = self.outerFrame.getTimes(dstr, en)
        res = times[1] - times[0]

        td = timedelta(seconds=val*res)
        self.timeLbl.setText(str(td))

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
        self.outerFrame.ui.radUnitBox.textChanged.connect(self.updtUnits)
        self.outerFrame.ui.radiusBox.valueChanged.connect(self.adjustScale)
        self.ui.magLineScaleBox.valueChanged.connect(self.saveScale)

        self.ui.modelBtn.clicked.connect(self.openMagTool)

        # Initialize the units label for mag line scaling
        self.updtUnits(self.outerFrame.getRadiusUnits())

        # Enable/disable relavent plot options when tick and plot types are changed
        for btn in self.ui.markerBtns:
            btn.toggled.connect(self.tickTypeChanged)

        for btn in [self.ui.projBtn, self.ui.viewPlaneBtn]:
            btn.toggled.connect(self.plotTypeChanged)

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
        self.tickTypeChanged()
        posDstrs = self.outerFrame.getPosVec()
        vecDstrs = self.outerFrame.getFieldVec()
        self.setVecScale(posDstrs, vecDstrs, a,b)
        self.scaleSet = True

    def getTickType(self):
        for btn in self.ui.markerBtns:
            if btn.isChecked():
                return btn.text()
        return 'None'

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
            self.enableLineOptions(False)
            self.enableIntervalOptions(True)
            # Update interval box value
            a, b = self.outerFrame.getIndices()
            self.ui.intervalBox.setValue(max(1, int((b-a)*.002)))
        else:
            self.tickTypeChanged()

        self.ui.btnBox.setEnabled(not projMode)

    def enableLineOptions(self, val=True):
        self.ui.centerLinesBox.setEnabled(val)
        self.ui.magLineScaleBox.setEnabled(val)
        self.ui.magLineLbl.setEnabled(val)

    def enableIntervalOptions(self, val=True):
        self.ui.intervalBox.setEnabled(val)
        self.ui.timeLbl.setEnabled(val)
        self.ui.intLbl.setEnabled(val)

    def tickTypeChanged(self):
        # Enable/disable UI elements according to tick type chosen
        tickType = self.getTickType()

        magScaleEnabled = True if tickType == 'Field Lines' else False
        tickInterval = False if tickType == 'None' else True

        self.enableLineOptions(magScaleEnabled)
        self.enableIntervalOptions(tickInterval)

        # Update tick spacing if switching between marker types
        val = None
        if tickInterval and tickType == 'Time Ticks':
            a, b = self.outerFrame.getIndices()
            val = int((b-a)/15)
        elif tickInterval:
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

    def getPosAndField(self, posDstrs, vecDstrs, sI, eI):
        # Get the plot's x,y coordinates in the plane chosen by user
        en = self.outerFrame.getEditNum()
        aAxisNum, bAxisNum = self.getAxisNum(1), self.getAxisNum(2)

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

    def plotTermProj(self, plt, pen, xDta, yDta, tickWidth):
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

    def plotTimeTicks(self, plt, pen, xDta, yDta, times, tickWidth):
        # Add round points at given orbit positions
        brush = pg.mkBrush(pen.color())
        plt.scatterPlot(xDta, yDta, size=tickWidth/4, pen=pen, brush=brush, pxMode=False)

        # Rotate text by 45 degrees if the average slope is 0.5
        yDiff = abs(np.diff(yDta))
        xDiff = abs(np.diff(xDta))
        avgSlope = np.mean(yDiff/xDiff)
        if avgSlope < 0.5:
            angle = 45
        else:
            angle = 0

        # Anchor text to the left of point if the orbit is close to
        # the origin and on its left side
        magDta = np.sqrt((xDta ** 2) + (yDta ** 2))
        startStr, endStr = ' ', ''
        anchor = (0, 0.5)
        minIndex = np.argmin(magDta)
        minDist = magDta[minIndex]
        minX = xDta[minIndex]
        originRadius = self.getOriginRadius(self.outerFrame.getPosVec())
        if (minDist < originRadius * 4 and minX < 0 and max) and angle == 0:
            anchor = (1, 0.5)
            endStr = ' '

        # Create a datetime axis to convert time ticks into timestamps
        tm = DateAxis(self.outerFrame.epoch, 'bottom')
        tm.tm.tO = times[0]
        tm.tm.tE = times[-1]

        # Add a timestamp text item near each of the plotted points,
        # with the given angle calculated above
        for x, y, t in zip(xDta, yDta, times):
            label = self.outerFrame.getTimestampFromTick(t)
            label = startStr + tm.fmtTimeStmp(label) + endStr
            txt = pg.TextItem(label, color=pg.mkColor('#000000'), anchor=anchor, angle=angle)
            plt.addItem(txt)
            txt.setPos(x, y)

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
        aAxisNum = self.getAxisNum(1)
        bAxisNum = self.getAxisNum(2)
        xcoords, ycoords, coords, tiltAngle = self.magTool.getFieldLines(aAxisNum, bAxisNum)

        # Plot each trace and print out the dipole tilt angle
        pen = pg.mkPen('#d47f00')
        pen.setWidthF(1.1)
        for x, y in zip(xcoords, ycoords):
            plt.plot(x,y, pen=pen)
        self.outerFrame.ui.statusBar.showMessage('Dipole Tilt Angle: '+str(tiltAngle))

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
        t0, t1 = times[0], times[-1] # Save start/end times for time label

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

        # Create plot item and initialize pens
        pen = self.outerFrame.getPens()[0]
        orbitPen = pg.mkPen(pen.color())
        plt = OrbitPlotItem()

        # Plot optional magnetosphere model field lines first (lowest level)
        if self.ui.modelBox.isChecked():
            self.plotMagnetosphere(plt, t0, xDta, yDta)

        # Limit data/times by the user's sample rate if plotting markers
        tickType = self.getTickType() if not projMode else 'Time Ticks'
        if tickType != 'None':
            rate = self.ui.intervalBox.value()
            xDta = self.sampleData(xDta, times, rate)
            yDta = self.sampleData(yDta, times, rate)
            xField = self.sampleData(xField, times, rate)
            yField = self.sampleData(yField, times, rate)
            times = self.sampleData(times, times, rate)

        # Mask out data points where an error flag is seen for any value
        mask = [True] * len(times)
        for arr in [xDta, yDta, xField, yField, times]:
            subMask = arr < self.outerFrame.window.errorFlag
            mask = mask & (subMask) # AND operation between masks

        xDta = xDta[mask]
        yDta = yDta[mask]
        xField = xField[mask]
        yField = yField[mask]
        times = times[mask]

        # Get tick marker width
        tickWidth = self.getTickWidth(xDta, yDta, tickType != 'None')

        # Plot orbit and its arrow to indicate direction
        if not projMode:
            gaps = self.outerFrame.getSegments(sI, eI) # Find gap indices as well
            plt.plot(xFull, yFull, pen=orbitPen, connect=gaps)
            arrowPen = pg.mkPen('#0a004a') if tickType == 'Field Lines' else orbitPen
            plt.plotArrowLine(xFull[-2:], yFull[-2:], tickWidth, arrowPen)

        # Add additional markers onto plot
        if projMode:
            self.plotTermProj(plt, pen, xDta, yDta, tickWidth)
        elif tickType == 'Field Lines':
            self.plotMagLines(plt, pen, xField, yField, xDta, yDta, tickWidth)
        elif tickType == 'Time Ticks':
            self.plotTimeTicks(plt, pen, xDta, yDta, times, tickWidth)

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
        te = TimeEdit(QtGui.QFont())
        self.refTimeBox = te.start
        te.setupMinMax(outerFrame.outerFrame.window.getMinAndMaxDateTime())

        # Coordinate system box
        self.coordBox = QtWidgets.QComboBox()
        self.coordBox.addItems(['GSM', 'GSE'])
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
        self.earthRadius = self.outerFrame.outerFrame.getEarthRadius()
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

    def updtTime(self):
        # Update time if it hasn't been set or if the set time is out of
        # the selected time range
        frameTime = self.outerFrame.outerFrame.ui.timeEdit.start.dateTime()
        frameEnd = self.outerFrame.outerFrame.ui.timeEdit.end.dateTime()
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
        gsmSystem = self.ui.coordBox.currentText() == 'GSM'
        rLim = self.ui.rlimBox.value()
        return dt, gsmSystem, rLim

    def getFieldLines(self, axisNum1, axisNum2):
        # Update time if selection is out of plot time range
        self.updtTime() 

        # Extract parameters and other plot settings
        self.earthRadius = self.outerFrame.outerFrame.getEarthRadius()
        dt, gsmSystem, rLim = self.getOtherParameters()
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
                    line = MagnetosphereTool.calcFieldLine(p, parmod, vecDir, rLim, r0)
                    for lst, axisLine in zip(coordLists, line):
                        lst.append(line)
        else:
            coordLists = self.multiProcFieldLines(plotCoords, parmod, r0, rLim)

        # Scale the coordinates if the plot units are in kilometers instead of RE
        self.scaleCoords(coordLists[0], coordLists[1], coordLists[2])
        plotCoords = [(x*self.earthRadius, y*self.earthRadius, z*self.earthRadius) for x,y,z in plotCoords]

        # Change coordinates if output is GSE
        if not gsmSystem:
            coordLists = self.coordsToGSE(coordLists[0], coordLists[1], coordLists[2])

        # Dipole tilt angle returned should be in degrees
        dipoleTilt = np.degrees(dipoleTilt)
        return coordLists[axisNum2], coordLists[axisNum1], plotCoords, dipoleTilt

    def multiProcFieldLines(self, plotCoords, parmod, r0, rLim):
        # Multiprocessed calculation of the field lines
        # Get number of lines to calculate per process and split coords into groups
        grpSize = int(len(plotCoords) / self.numThreads)
        grps = []
        lastIndex = 0
        for startIndex in range(0, len(plotCoords), grpSize):
            parmList = (parmod[:], rLim, r0)
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
        xGSE, yGSE, zGSE = [], [], []
        for lineX, lineY, lineZ in zip(xCoords, yCoords, zCoords):
            newX, newY, newZ = [], [], []
            for x, y, z in zip(lineX, lineY, lineZ):
                xgse, ygse, zgse = geopack.gsmgse(x,y,z,100)
                newX.append(xgse)
                newY.append(ygse)
                newZ.append(zgse)
            xGSE.append(newX)
            yGSE.append(newY)
            zGSE.append(newZ)
        return [xGSE, yGSE, zGSE]

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
        parmod, rLim, r0 = parmList
        # Get line at each coordinate in both directions
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

class OrbitPlotItem(MagPyPlotItem):
    def __init__(self):
        self.originItem = None
        self.scaleBar = None

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
