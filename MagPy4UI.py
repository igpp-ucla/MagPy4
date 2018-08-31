
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
import functools

from pyqtgraphExtensions import GridGraphicsLayout,LinearGraphicsLayout,BLabelItem
from mth import Mth

class MagPy4UI(object):

    def buildPopup(self, name, actions):
        # add options popup menu for toggled things
        popup = QtWidgets.QToolButton()
        menu = QtWidgets.QMenu()
        menu.setToolTipsVisible(True)
        for action in actions:
            menu.addAction(action)

        popup.setMenu(menu)
        popup.setText(f'{name} ') # extra space for little arrow icon
        popup.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        return popup 

    def setupUI(self, window):

        # gives default window options in top right
        window.setWindowFlags(QtCore.Qt.Window)
        window.resize(1280,700)

        self.centralWidget = QtWidgets.QWidget(window)
        window.setCentralWidget(self.centralWidget)

        # define actions
        self.actionOpenFF = QtWidgets.QAction(window)
        self.actionOpenFF.setPriority(QtWidgets.QAction.HighPriority)
        self.actionOpenFF.setText('Open FF')
        self.actionOpenFF.setShortcut('O')
        self.actionOpenFF.setToolTip('Opens a flatfile (requires valid .ffd and .ffh files)')

        self.actionAddFF = QtWidgets.QAction(window)
        self.actionAddFF.setPriority(QtWidgets.QAction.HighPriority)
        self.actionAddFF.setText('Add FF')
        self.actionAddFF.setToolTip('Adds a flatfile to current (requires valid .ffd and .ffh files)')

        self.actionOpenCDF = QtWidgets.QAction(window)
        self.actionOpenCDF.setPriority(QtWidgets.QAction.HighPriority)
        self.actionOpenCDF.setText('Open CDF')
        self.actionOpenCDF.setToolTip('Opens a .cdf file (experimental)')

        self.actionShowData = QtWidgets.QAction(window)
        self.actionShowData.setText('Data')
        self.actionShowData.setToolTip('Shows the loaded data in a table view')

        self.actionPlot = QtWidgets.QAction(window)
        self.actionPlot.setText('Plot')
        self.actionPlot.setToolTip('Opens plot menu')

        self.actionSpectra = QtWidgets.QAction(window)
        self.actionSpectra.setText('Spectra')
        self.actionSpectra.setToolTip('Opens spectral analysis window')

        self.actionEdit = QtWidgets.QAction(window)
        self.actionEdit.setText('Edit')
        self.actionEdit.setToolTip('Opens edit window that allows you to rotate the data with matrices')

        self.scaleYToCurrentTimeAction = QtWidgets.QAction('Scale y range to current time selection',checkable=True,checked=True)
        self.antialiasAction = QtWidgets.QAction('Smooth lines (antialiasing)',checkable=True,checked=True)
        self.bridgeDataGaps = QtWidgets.QAction('Bridge Data Gaps', checkable=True, checked=False)
        self.drawPoints = QtWidgets.QAction('Draw Points (unoptimized)', checkable=True, checked=False)

        self.runTests = QtWidgets.QAction(window)
        self.runTests.setText('Run Tests')
        self.runTests.setToolTip('Runs unit tests for code')

        self.switchMode = QtWidgets.QAction(window)
        #self.switchMode.setText('Switch to MarsPy')
        #self.switchMode.setToolTip('Loads various presets specific to the Insight mission')

        # build toolbar (todo switch to QMenuBar class, thats what i should have used to begin with lol
        self.toolBar = QtWidgets.QToolBar(window)
        window.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)

        file = self.buildPopup('File', [self.actionOpenFF, self.actionAddFF, self.actionOpenCDF])
        self.toolBar.addWidget(file)

        self.toolBar.addAction(self.actionShowData)
        self.toolBar.addAction(self.actionPlot)

        self.toolBar.addAction(self.actionSpectra)
        self.toolBar.addAction(self.actionEdit)

        options = self.buildPopup('Options', [self.scaleYToCurrentTimeAction, self.antialiasAction, self.bridgeDataGaps, self.drawPoints])
        self.toolBar.addWidget(options) 

        #empty widget (cant use spacer in toolbar?) does same thing tho so this action goes far right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.toolBar.addWidget(spacer)

        self.toolBar.addAction(self.switchMode)
        #self.toolBar.addAction(self.runTests)

        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout(window) # made this based off pg.GraphicsLayout
        #self.glw.setContentsMargins(0,0,0,0)
        self.timeLabel = BLabelItem()
        self.gview.sceneObj.addItem(self.timeLabel)

        self.gview.setCentralItem(self.glw)

        layout = QtWidgets.QVBoxLayout(self.centralWidget)
        layout.addWidget(self.gview)

        # SLIDER setup
        sliderLayout = QtWidgets.QGridLayout() # r, c, w, h
        self.startSlider = QtWidgets.QSlider()
        self.startSlider.setOrientation(QtCore.Qt.Horizontal)
        self.endSlider = QtWidgets.QSlider()
        self.endSlider.setOrientation(QtCore.Qt.Horizontal)

        self.timeEdit = TimeEdit(QtGui.QFont("monospace", 14))

        sliderLayout.addWidget(self.timeEdit.start, 0, 0, 1, 1)
        sliderLayout.addWidget(self.startSlider, 0, 1, 1, 1)
        sliderLayout.addWidget(self.timeEdit.end, 1, 0, 1, 1)
        sliderLayout.addWidget(self.endSlider, 1, 1, 1, 1)

        layout.addLayout(sliderLayout)

         # update slider tick amount and timers and labels and stuff based on new file
    def setupSliders(self, tick, max, minmax):
        #dont want to trigger callbacks from first plot
        self.startSlider.blockSignals(True)
        self.endSlider.blockSignals(True)

        self.startSlider.setMinimum(0)
        self.startSlider.setMaximum(max)
        self.startSlider.setTickInterval(tick)
        self.startSlider.setSingleStep(tick)
        self.startSlider.setValue(0)
        self.endSlider.setMinimum(0)
        self.endSlider.setMaximum(max)
        self.endSlider.setTickInterval(tick)
        self.endSlider.setSingleStep(tick)
        self.endSlider.setValue(max)

        self.timeEdit.setupMinMax(minmax)

        self.startSlider.blockSignals(False)
        self.endSlider.blockSignals(False)



class TimeEdit():
    def __init__(self, font):
        self.start = QtWidgets.QDateTimeEdit()
        self.end = QtWidgets.QDateTimeEdit()
        self.start.setFont(font)
        self.end.setFont(font)
        self.start.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.end.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        #self.start.setToolTip('Start Time'); # not always true if they get reversed...
        #self.end.setToolTip('End Time');
        self.start.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")
        self.end.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")
        self.start.editingFinished.connect(functools.partial(self.enforceMinMax, self.start))
        self.end.editingFinished.connect(functools.partial(self.enforceMinMax, self.end))

    def setupMinMax(self, minmax):
        min,max = minmax
        self.minDateTime = min
        self.maxDateTime = max
        self.setStartNoCallback(min)
        self.setEndNoCallback(max)

    def setWithNoCallback(dte, dt):
        dte.blockSignals(True)
        dte.setDateTime(dt)
        dte.blockSignals(False)

    def setStartNoCallback(self, dt):
        TimeEdit.setWithNoCallback(self.start, dt)

    def setEndNoCallback(self, dt):
        TimeEdit.setWithNoCallback(self.end, dt)

    # done this way to avoid mid editing corrections
    def enforceMinMax(self, dte):
        min = self.minDateTime
        max = self.maxDateTime
        dt = dte.dateTime()
        dte.setDateTime(min if dt < min else max if dt > max else dt)

    def toString(self):
        #form = "yyyy MM dd hh mm ss zzz"
        form = "yyyy MMM dd hh:mm:ss.zzz"
        d0 = self.start.dateTime().toString(form)
        d1 = self.end.dateTime().toString(form)
        return d0,d1    

class MatrixWidget(QtWidgets.QWidget):
    def __init__(self, type='labels', parent=None):
        #QtWidgets.QWidget.__init__(self, parent)
        super(MatrixWidget, self).__init__(parent)
        grid = QtWidgets.QGridLayout(self)
        self.mat = [] # matrix of label or line widgets
        grid.setContentsMargins(0,0,0,0)
        for y in Mth.i:
            row = []
            for x in Mth.i:
                if type == 'labels':
                    w = QtGui.QLabel('0.0')
                    w.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                elif type == 'lines':
                    w = QtGui.QLineEdit()
                    w.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly) #i dont even know if this does anything
                    w.setText('0.0')
                else:
                    assert False, 'unknown type requested in MatrixWidget!'
                grid.addWidget(w, y, x, 1, 1)
                row.append(w)
            self.mat.append(row)

        #self.update()

    def setMatrix(self, m):
        for i in Mth.i:
            for j in Mth.i:
                self.mat[i][j].setText(Mth.formatNumber(m[i][j]))
                self.mat[i][j].repaint() # mac doesnt repaint sometimes

    # returns list of numbers
    def getMatrix(self):
        M = Mth.empty()
        for i in Mth.i:
            for j in Mth.i:
                s = self.mat[i][j].text()
                try:
                    f = float(s)
                except ValueError:
                    print(f'matrix has non-number at location {i},{j}')
                    f = 0.0
                M[i][j] = f
        return M

    def toString(self):
        return Mth.matToString(self.getMatrix())

# pyqt utils
class PyQtUtils:
    def clearLayout(layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()

    def moveToFront(window):
        if window:
            # this will remove minimized status 
            # and restore window with keeping maximized/normal state
            #window.setWindowState(window.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
            window.raise_()
            # this will activate the window
            window.activateWindow()