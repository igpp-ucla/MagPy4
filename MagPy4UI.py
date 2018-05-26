
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg
import functools

from pyqtgraphExtensions import GridGraphicsLayout,LinearGraphicsLayout,BLabelItem

#from qrangeslider import QRangeSlider
#can add in translation stuff later

class MagPy4UI(object):
    def setupUI(self, MagPy4):

        # gives default window options in top right
        MagPy4.setWindowFlags(QtCore.Qt.Window)
        MagPy4.resize(1280,800)

        self.centralWidget = QtWidgets.QWidget(MagPy4)
        MagPy4.setCentralWidget(self.centralWidget)

        self.toolBar = QtWidgets.QToolBar(MagPy4)
        MagPy4.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpenFF = QtWidgets.QAction(MagPy4)
        self.actionOpenFF.setPriority(QtWidgets.QAction.HighPriority)
        self.actionOpenFF.setText('Open FF')
        self.actionOpenFF.setShortcut('O')
        self.toolBar.addAction(self.actionOpenFF)

        #self.actionOpenCDF = QtWidgets.QAction(MagPy4)
        #self.actionOpenCDF.setPriority(QtWidgets.QAction.HighPriority)
        #self.actionOpenCDF.setText('Open CDF')
        #self.toolBar.addAction(self.actionOpenCDF)

        self.actionShowData = QtWidgets.QAction(MagPy4)
        self.actionShowData.setText('Show Data')
        self.toolBar.addAction(self.actionShowData)

        self.actionPlot = QtWidgets.QAction(MagPy4)
        self.actionPlot.setText('Plot Menu')
        self.toolBar.addAction(self.actionPlot)

        self.actionSpectra = QtWidgets.QAction(MagPy4)
        self.actionSpectra.setText('Spectra')
        self.toolBar.addAction(self.actionSpectra)

        self.actionEdit = QtWidgets.QAction(MagPy4)
        self.actionEdit.setText('Edit')
        self.toolBar.addAction(self.actionEdit)

        # add options popup menu for toggled things
        options = QtWidgets.QToolButton()
        menu = QtWidgets.QMenu()

        self.scaleYToCurrentTimeAction = QtWidgets.QAction('Scale y range to current time selection',checkable=True,checked=True)
        self.antialiasAction = QtWidgets.QAction('Smooth lines (antialiasing)',checkable=True,checked=True)
        self.bridgeDataGaps = QtWidgets.QAction('Bridge Data Gaps', checkable=True, checked=False)
        self.drawPoints = QtWidgets.QAction('Draw Points', checkable=True, checked=False)

        menu.addAction(self.scaleYToCurrentTimeAction)
        menu.addAction(self.antialiasAction)
        menu.addAction(self.bridgeDataGaps)
        menu.addAction(self.drawPoints)
        options.setMenu(menu)
        options.setText('Options ') # extra space for little arrow icon
        options.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.toolBar.addWidget(options)        

        #empty widget (cant use spacer in toolbar?) does same thing tho so this action goes far right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.toolBar.addWidget(spacer)

        self.switchMode = QtWidgets.QAction(MagPy4)
        self.switchMode.setText('Switch to MarsPy')
        self.toolBar.addAction(self.switchMode)

        self.gview = pg.GraphicsView()
        self.gview.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.glw = GridGraphicsLayout() # made this based off pg.GraphicsLayout
        #self.glw.setContentsMargins(0,0,0,0)
        self.timeLabel = BLabelItem()
        self.gview.sceneObj.addItem(self.timeLabel)

        self.gview.setCentralItem(self.glw)

        layout = QtWidgets.QVBoxLayout(self.centralWidget)
        layout.addWidget(self.gview)

        # SLIDER setup
        sliderFont = QtGui.QFont("monospace", 14)#, QtGui.QFont.Bold) 
        sliderLayout = QtWidgets.QGridLayout() # r, c, w, h
        self.startSlider = QtWidgets.QSlider()
        self.startSlider.setOrientation(QtCore.Qt.Horizontal)
        self.endSlider = QtWidgets.QSlider()
        self.endSlider.setOrientation(QtCore.Qt.Horizontal)

        self.startSliderEdit = QtWidgets.QDateTimeEdit()
        self.endSliderEdit = QtWidgets.QDateTimeEdit()
        self.startSliderEdit.setFont(sliderFont)
        self.endSliderEdit.setFont(sliderFont)
        self.startSliderEdit.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")
        self.endSliderEdit.setDisplayFormat("yyyy MMM dd hh:mm:ss.zzz")

        sliderLayout.addWidget(self.startSliderEdit, 0, 0, 1, 1)
        sliderLayout.addWidget(self.startSlider, 0, 1, 1, 1)
        sliderLayout.addWidget(self.endSliderEdit, 1, 0, 1, 1)
        sliderLayout.addWidget(self.endSlider, 1, 1, 1, 1)

        layout.addLayout(sliderLayout)

        self.sliderMinDateTime = None
        self.sliderMaxDateTime = None
        self.startSliderEdit.editingFinished.connect(functools.partial(self.enforceMinMax, self.startSliderEdit))
        self.endSliderEdit.editingFinished.connect(functools.partial(self.enforceMinMax, self.endSliderEdit))

         # update slider tick amount and timers and labels and stuff based on new file
    def setupSliders(self, tick, max, minDateTime, maxDateTime):
        #dont want to trigger callbacks from first plot
        self.startSlider.blockSignals(True)
        self.endSlider.blockSignals(True)
        self.startSliderEdit.blockSignals(True)
        self.endSliderEdit.blockSignals(True)

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

        self.sliderMinDateTime = minDateTime
        self.sliderMaxDateTime = maxDateTime
        self.startSliderEdit.setDateTime(minDateTime)
        self.endSliderEdit.setDateTime(maxDateTime)

        self.startSlider.blockSignals(False)
        self.endSlider.blockSignals(False)
        self.startSliderEdit.blockSignals(False)
        self.endSliderEdit.blockSignals(False)

    def enforceMinMax(self, dte):
        min = self.sliderMinDateTime
        max = self.sliderMaxDateTime
        dt = dte.dateTime()
        dte.setDateTime(min if dt < min else max if dt > max else dt)