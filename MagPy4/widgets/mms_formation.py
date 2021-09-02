from ..layouttools import BaseLayout
from ..MagPy4UI import TimeEdit
from .mmstools import MMSTools
from ..plotbase import MagPyPlotItem
from .mms_orbit import MMS_Data_Tool

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import numpy as np
import bisect
from datetime import datetime, date
import pyqtgraph as pg
import os

class MMS_FormationUI(BaseLayout):
    def setupUI(self, frame):
        self.frame = frame
        frame.resize(300, 100)
        frame.move(100, 100)
        frame.setWindowTitle('MMS Spacecraft Formation')
        self.layout = QtWidgets.QGridLayout(frame)

        # Set up settings layout
        self.settingsLt = self.setupSettingsLt()
        self.layout.addWidget(self.settingsLt, 0, 0, 1, 1)

        # Status bar
        self.statusBar = QtWidgets.QStatusBar()
        self.statusBar.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
        self.layout.addWidget(self.statusBar, 1, 0, 1, 1)

        # Variable indicating which plot type mode is active
        self.currentMode = None

    def getCurrentMode(self):
        # Returns the plotting layout currently being used
        return self.currentMode

    def getCanvasFrm(self, frame):
        # Set up canvas for drawing matplotlib figures
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)

        # Set up figure/canvas
        figsize = (6, 5.5)
        if os.name not in ['posix', 'nt']:
            figsize = (3, 3)
        self.figure = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)

        # Set up toolbar
        self.canvasToolbar = NavigationToolbar(self.canvas, frame)
        self.canvasToolbar.setIconSize(QtCore.QSize(20, 20))

        # Add items to layout
        layout.addWidget(self.canvas, 0, 0, 1, 1)
        layout.addWidget(self.canvasToolbar, 1, 0, 1, 1)

        # Set size policy for canvas
        item = layout.itemAtPosition(0, 0).widget()
        item.setSizePolicy(self.getSizePolicy('Min', 'Min'))

        return frame

    def getProjFrm(self):
        # Sets up grid graphics layout to plot projections of spacecraft formation
        frame = QtWidgets.QFrame()
        layout = QtWidgets.QGridLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)

        # Set up graphics grid
        self.glw = self.getGraphicsGrid()

        # Set minimum size
        self.glw.setSizePolicy(self.getSizePolicy('Min', 'Min'))
        self.gview.setMinimumHeight(550)
        self.gview.setMinimumWidth(550)

        layout.addWidget(self.gview, 0, 0, 1, 1)
        return frame

    def plotTypeChanged(self, plotType):
        # Sets whether 'View' options box for projection-type plots is visible
        val = True
        if plotType == '3D':
            val = False

        self.viewPairLbl.setVisible(val)
        self.viewPairBox.setVisible(val)

    def timeModeChanged(self, mode):
        # Sets whether both time edits are visible or not + sets the label accordingly
        lbl = 'Start Time: '
        startLbl, endLbl = self.timeLbls
        endVisible = True

        if mode == 'Reference Time':
            lbl = 'Time: '
            endVisible = False

        startLbl.setText(lbl)
        self.timeEdit.end.setVisible(endVisible)
        endLbl.setVisible(endVisible)

    def setupSettingsLt(self):
        # Wrapper layout includes update button, regular layout encapsulates
        # all settings boxes/options
        wrapperFrame = QtWidgets.QFrame()
        wrapperLt = QtWidgets.QVBoxLayout(wrapperFrame)
        wrapperLt.setContentsMargins(0, 0, 0, 0)

        frame = QtWidgets.QGroupBox('Settings')
        layout = QtWidgets.QGridLayout(frame)

        # Set up plot type selection layout
        pltTypeLt = QtWidgets.QHBoxLayout()
        pltTypeLbl = QtWidgets.QLabel('Plot Type: ')
        pltTypeLbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.plotTypeBox = QtWidgets.QComboBox()
        self.plotTypeBox.addItems(['3D', 'Projection'])
        pltTypeLt.addWidget(pltTypeLbl)
        pltTypeLt.addWidget(self.plotTypeBox)

        # Set up view pair layout
        viewPairLt = QtWidgets.QHBoxLayout()
        self.viewPairBox = QtWidgets.QComboBox()
        self.viewPairBox.addItems(['XY', 'YZ', 'XZ'])
        self.viewPairLbl = QtWidgets.QLabel(' View: ')

        viewPairLt.addWidget(self.viewPairLbl)
        viewPairLt.addWidget(self.viewPairBox)

        # Add view pair layout to plot type layout
        pltTypeLt.addLayout(viewPairLt)

        # Connect changes in plot type to setting viewPairLt visible/hidden
        self.plotTypeBox.currentTextChanged.connect(self.plotTypeChanged)
        self.plotTypeChanged(self.plotTypeBox.currentText())

        # Time mode layout
        modeLt = QtWidgets.QHBoxLayout()
        modeLbl = QtWidgets.QLabel('Mode: ')
        modeLbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.modeBox = QtWidgets.QComboBox()
        self.modeBox.addItems(['Time-Averaged', 'Reference Time'])
        modeLt.addWidget(modeLbl)
        modeLt.addWidget(self.modeBox)

        # Set up start/end time layout
        timeLt = QtWidgets.QGridLayout()
        self.timeEdit = TimeEdit()
        l1 = self.addPair(timeLt, 'Start Time: ', self.timeEdit.start, 0, 0, 1, 1)
        l2 = self.addPair(timeLt, 'End Time: ', self.timeEdit.end, 1, 0, 1, 1)
        l1.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.timeEdit.start.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum))
        self.timeLbls = (l1, l2)

        # Connect changes in mode to setting end time hidden/visible
        self.modeBox.currentTextChanged.connect(self.timeModeChanged)
        self.timeModeChanged(self.modeBox.currentText())

        # Set up scale layout
        scaleLt = QtWidgets.QGridLayout()
        scaleLbl = QtWidgets.QLabel('Units: ')
        scaleLbl.setSizePolicy(self.getSizePolicy('Max', 'Max'))
        self.scaleBox = QtWidgets.QComboBox()
        self.scaleBox.addItems(['KM', 'RE'])
        scaleLt.addWidget(scaleLbl, 0, 0, 1, 1)
        scaleLt.addWidget(self.scaleBox, 0, 1, 1, 1)

        # Plot barycenter check
        optionsLt = QtWidgets.QGridLayout()
        self.plotBarycenter = QtWidgets.QCheckBox(' Plot barycenter')
        optionsLt.addWidget(self.plotBarycenter)

        # Update button
        updtLt = QtWidgets.QHBoxLayout()
        self.updateBtn = QtWidgets.QPushButton('  Update  ')
        updtLt.addStretch()
        updtLt.addWidget(self.updateBtn)

        # Set up layouts
        layout.addLayout(pltTypeLt, 0, 0, 1, 1)
        layout.addLayout(modeLt, 1, 0, 1, 1)
        layout.addLayout(timeLt, 2, 0, 1, 1)
        layout.addLayout(scaleLt, 3, 0, 1, 1)
        layout.addLayout(optionsLt, 4, 0, 1, 1)

        wrapperLt.addWidget(frame)
        wrapperLt.addLayout(updtLt)
        wrapperLt.addStretch()

        return wrapperFrame

    def switchMode(self):
        # Get the current plot type and build the layout
        val = self.plotTypeBox.currentText()
        if val == '3D':
            frm = self.getCanvasFrm(self.frame)
        else:
            frm = self.getProjFrm()

        # Remove old plot layout if there is any
        oldFrm = self.layout.itemAtPosition(0, 1)
        if oldFrm is not None:
            self.layout.removeItem(oldFrm)
            oldFrm.widget().deleteLater()

        # Add new layout to outer frame's layout
        self.layout.addWidget(frm, 0, 1, 2, 1)

class MMS_Formation(QtWidgets.QFrame, MMS_FormationUI, MMSTools):
    def __init__(self, window, parent=None):
        self.window = window
        super().__init__(parent)
        MMSTools.__init__(self, window)

        self.ui = MMS_FormationUI()
        self.ui.setupUI(self)

        self.ui.updateBtn.clicked.connect(self.updatePlot)

        # Colors/pairs constants
        self.colors = self.window.mmsColors[:]
        self.pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        self.earthRadius = 6371.2

        # Save time-averaged position data to use if time range not updated
        self.cachedData = None

        # Set up min/max values for time edit
        minDt = datetime(2015, 4, 1, 1)
        currDt = date.today()
        maxDt = datetime(currDt.year, currDt.month, currDt.day)
        self.ui.timeEdit.setupMinMax((minDt, maxDt))

        # Set time edit values to time range of loaded data
        startDt, endDt = self.window.getMinAndMaxDateTime()
        self.ui.timeEdit.start.setDateTime(startDt)
        self.ui.timeEdit.end.setDateTime(endDt)

    def getStartEndTime(self):
        startDt = self.ui.timeEdit.start.dateTime().toPyDateTime()
        endDt = self.ui.timeEdit.end.dateTime().toPyDateTime()

        # Use start date only if not using time-averaged data
        mode = self.ui.modeBox.currentText()
        if mode == 'Reference Time':
            endDt = startDt

        return min(startDt, endDt), max(startDt, endDt)

    def _tickInRange(self, refTick, startTick, endTick):
        # Round ranges to 3 decimal places to work properly with ticks
        # mapped from time edits
        startTick = min(np.around(startTick, 3), startTick)
        endTick = np.around(endTick, 3)
        refTick = np.around(refTick, 3)
        if refTick >= startTick and refTick <= endTick:
            return True

    def getPlotData(self, units='KM'):
        # Map start/end time to indices
        startDt, endDt = self.getStartEndTime()
        startTick = self.window.getTickFromDateTime(startDt)
        endTick = self.window.getTickFromDateTime(endDt)

        # Check if current time range is same as before, returning the previous
        # result if so
        if self.cachedData is not None:
            (tO, tE), posDta = self.cachedData
            if startTick == tO and endTick == tE:
                if units == 'RE':
                    posDta = [dta/self.earthRadius for dta in posDta]
                return posDta

        # Compare time range to time range in loaded data
        arbDstr = self.getDstrsBySpcrft(1, grp='Pos')[0]
        times = self.window.getTimes(arbDstr, 0)[0]
        tO, tE = times[0], times[-1]

        # Get averaged position data from loaded data
        if self._tickInRange(startTick, tO, tE) and self._tickInRange(endTick, tO, tE):
            # Find start/end indices
            startIndex = bisect.bisect(times, startTick)
            endIndex = bisect.bisect(times, endTick)

            # Get averaged position vector for each spacecraft
            posDta = []
            for scNum in [1, 2, 3, 4]:
                data = self.getPosData(scNum, startIndex, endIndex+1)
                avgPos = np.mean(data, axis=1)
                posDta.append(avgPos)

        else: # Otherwise, download data from LASP
            self.ui.statusBar.showMessage('Downloading MMS position data...')
            self.ui.processEvents()

            scDta, scTimes = MMS_Data_Tool.getPosDta((startDt, endDt), [1,2,3,4])
            scDta = {scNum : scDta[scNum]['gsm'] for scNum in [1,2,3,4]}

            # Apply similar averaging operation to dict of pos data as above
            posDta = []
            for scNum in [1,2,3,4]:
                # Get start/end indices for each spacecraft data
                times = scTimes[scNum]
                startIndex = bisect.bisect(times, startTick)
                endIndex = bisect.bisect(times, endTick)

                # Get averaged position vector
                data = scDta[scNum][:,startIndex:endIndex+1]
                avgPos = np.mean(data, axis=1)
                posDta.append(avgPos)
        self.ui.statusBar.clearMessage()

        # Store calculated time averaged position data
        timeRng = (startTick, endTick)
        self.cachedData = (timeRng, posDta)

        # Adjust units
        if units == 'RE':
            posDta = [dta/self.earthRadius for dta in posDta]

        return posDta

    def getBarycenter(self, posDta):
        barycenterVec = np.mean(posDta, axis=0)
        return barycenterVec

    def getTitle(self, mode):
        title = 'MMS Formation'
        if mode == 'Time-Averaged':
            title = title + ' (Time-Averaged)'

        return title

    def plot3D(self):
        # Clear previous figure
        self.ui.figure.clf()

        # Get averaged position data and scale according to units
        units = self.ui.scaleBox.currentText()
        mode = self.ui.modeBox.currentText()
        posDta = self.getPlotData(units)

        xDta = [x for x, y, z in posDta]
        yDta = [y for x, y, z in posDta]
        zDta = [z for x, y, z in posDta]

        # Set up figure
        ax = self.ui.figure.add_subplot(111, projection='3d')

        ## Set axis labels
        coordSys = 'GSM'
        funs = [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel]
        axes = ['X', 'Y', 'Z']
        for f, axis in zip(funs, axes):
            axisLabel = f'{axis} {coordSys} ({units})'
            f(axisLabel, labelpad=10)

        ## Set plot title
        title = self.getTitle(mode)
        self.ui.figure.suptitle(title)

        ## Adjust whitespace
        ax.margins(0.1, 0.1, 0.1, tight=True)
        self.ui.figure.tight_layout()

        # Plot points
        pts = ax.scatter(xDta, yDta, zDta, s=125, c=self.colors, depthshade=False)

        # Plot lines
        for a, b in self.pairs:
            xProj = [xDta[a-1], xDta[b-1]]
            yProj = [yDta[a-1], yDta[b-1]]
            zProj = [zDta[a-1], zDta[b-1]]
            ax.plot(xProj, yProj, zProj, c='#000000')

        # Plot barycenter
        if self.ui.plotBarycenter.isChecked():
            xc, yc, zc = self.getBarycenter(posDta)
            ax.scatter([xc], [yc], [zc], s=125, c='#474747', marker='+')

            # Add text label
            scaleFactor = self.earthRadius if units == 'KM' else 1
            lbls = [str(np.around(v/scaleFactor, 3)) for v in [xc, yc, zc]]
            lblStr = ', '.join(lbls) + ' RE\n'
            fontDict = {'horizontalalignment':'center', 'verticalalignment':'bottom'}
            ax.text(xc, yc, zc, lblStr, fontdict=fontDict)

        # Add items to legend
        handles = []
        for scNum, color in zip([1,2,3,4], self.colors):
            tmp = mpatches.Circle((0, 0), color=color, label=f'MMS {scNum}')
            handles.append(tmp)
        ax.legend(handles=handles, loc='upper left', handlelength=1)

        # Update plot
        self.ui.canvas.draw()

    def getTimeLabel(self, mode):
        # Maps start/end dates to timestamps
        startDt, endDt = self.getStartEndTime()
        fmtStr = '%Y %b %d %H:%M:%S.%f'
        startTs = startDt.strftime(fmtStr)[:-3]
        endTs = endDt.strftime(fmtStr)[:-3]

        if mode == 'Time-Averaged':
            lbl = f'Time Range: {startTs} to {endTs}'
        else:
            lbl = f'Time: {startTs}'

        return lbl

    def plotProj(self):
        # Clear previous plot
        self.ui.glw.clear()

        # Get time-averaged position data
        units = self.ui.scaleBox.currentText()
        mode = self.ui.modeBox.currentText()
        viewPair = self.ui.viewPairBox.currentText()

        posDta = self.getPlotData(units)
        axisMap = {'X':0, 'Y':1, 'Z':2}
        axisLeft = axisMap[viewPair[1]]
        axisBottom = axisMap[viewPair[0]]

        ## Map to projection data
        xDta = [v[axisBottom] for v in posDta]
        yDta = [v[axisLeft] for v in posDta]

        # Set up plot item
        plt = MagPyPlotItem()
        plt.hideButtons()
        for axisName in ['right', 'top']:
            plt.showAxis(axisName)
            plt.getAxis(axisName).setStyle(showValues=False)

        ## Lock aspect ratio
        plt.setAspectLocked(True, 1.0)

        ## Set axis labels and title
        plt.getAxis('left').setLabel(f'{viewPair[1]} GSM ({units})')
        plt.getAxis('bottom').setLabel(f'{viewPair[0]} GSM ({units})')
        title = self.getTitle(mode)
        plt.setTitle(title, size='13pt')

        # Draw points
        pens = [pg.mkPen(color) for color in self.colors]
        brushes = [pg.mkBrush(color) for color in self.colors]
        plt.scatterPlot(xDta, yDta, pxMode=True, size=24, pen=pens, brush=brushes)

        # Add text labels to indicate which spacecraft each point represents
        for x, y, scNum in zip(xDta, yDta, [1,2,3,4]):
            textItem = pg.TextItem(f'{scNum}', anchor=(0.5, 0.5), color=(255, 255, 255))
            plt.addItem(textItem)
            textItem.setPos(x, y)

        # Draw lines between spacecraft positions
        pen = pg.mkPen('#000000')
        pen.setWidthF(1.5)
        for a, b in self.pairs:
            x = [xDta[a-1], xDta[b-1]]
            y = [yDta[a-1], yDta[b-1]]
            pdi = plt.plot(x, y, pen=pen)
            pdi.setZValue(-100)

        # Draw barycenter
        if self.ui.plotBarycenter.isChecked():
            centerVec = self.getBarycenter(posDta)
            x, y = centerVec[axisBottom], centerVec[axisLeft]

            # Create scatter plot object
            penColor = '#474747'
            pen = pg.mkPen(penColor)
            brush = pg.mkBrush(penColor)
            plt.scatterPlot([x], [y], pen=pen, brush=brush, pxMode=True, 
                symbol='+', size=12)

            # Draw text label near it
            scaleFactor = self.earthRadius if units == 'KM' else 1
            xLbl = np.around(x/scaleFactor, 3)
            yLbl = np.around(y/scaleFactor, 3)
            lbl = f'{xLbl}, {yLbl} RE\n'
            lbl = pg.TextItem(lbl, anchor=(0.5, 0.75), color='#000000')
            lbl.setPos(x, y)
            plt.addItem(lbl)

        # Create time label
        lbl = pg.LabelItem(self.getTimeLabel(mode))
        lbl.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred))

        # Add to grid layout
        self.ui.glw.addItem(plt, 0, 0, 1, 1)
        self.ui.glw.addItem(lbl, 1, 0, 1, 1)

    def updatePlot(self):
        plotType = self.ui.plotTypeBox.currentText()

        # Update layout if plot type switched
        if plotType != self.ui.getCurrentMode():
            self.ui.switchMode()

        if plotType == '3D':
            self.plot3D()
        else:
            self.plotProj()