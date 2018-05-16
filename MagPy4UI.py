
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg

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

        self.actionPlot = QtWidgets.QAction(MagPy4)
        self.actionPlot.setText('Plot Menu')
        self.toolBar.addAction(self.actionPlot)

        self.actionShowData = QtWidgets.QAction(MagPy4)
        self.actionShowData.setText('Show Data')
        self.toolBar.addAction(self.actionShowData)

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

        self.glw = pg.GraphicsLayoutWidget()#border=(100,100,100))
        self.glw.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        
        layout = QtWidgets.QVBoxLayout(self.centralWidget)
        layout.addWidget(self.glw)
        #layout.setContentsMargins(0,0,0,0)

        # SLIDER setup
        sliderFont = QtGui.QFont("Times", 14)#, QtGui.QFont.Bold) 
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
