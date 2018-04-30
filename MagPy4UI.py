
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
        self.actionOpen = QtWidgets.QAction(MagPy4)
        self.actionOpen.setPriority(QtWidgets.QAction.HighPriority)
        self.actionOpen.setText('Open')
        self.actionOpen.setShortcut('O')

        self.actionPlot = QtWidgets.QAction(MagPy4)
        self.actionPlot.setText('Plot Menu')

        self.actionShowData = QtWidgets.QAction(MagPy4)
        self.actionShowData.setText('Show Data')

        self.switchMode = QtWidgets.QAction(MagPy4)
        self.switchMode.setText('Switch to MarsPy')

        self.actionSpectra = QtWidgets.QAction(MagPy4)
        self.actionSpectra.setText('Spectra')

        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionPlot)
        self.toolBar.addAction(self.actionShowData)
        self.toolBar.addAction(self.actionSpectra)
        self.toolBar.addAction(self.switchMode)

        self.options = QtWidgets.QToolButton()
        
        self.menu = QtWidgets.QMenu()
        self.scaleYToCurrentTimeAction = QtWidgets.QAction('Scale y range to current time selection',checkable=True,checked=True)
        self.antialiasAction = QtWidgets.QAction('Smooth lines (antialiasing)',checkable=True,checked=True)
        self.test3 = QtWidgets.QAction('test3',checkable=True)
        self.menu.addAction(self.scaleYToCurrentTimeAction)
        self.menu.addAction(self.antialiasAction)
        self.menu.addAction(self.test3)

        self.options.setMenu(self.menu)
        self.options.setText('Options ')
        self.options.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.toolBar.addWidget(self.options)
        

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

        #optionsLayout = QtWidgets.QHBoxLayout()
        #scaleLabel = QtWidgets.QLabel()
        #scaleLabel.setText("Scale Y range to current time selection")
        #self.scaleYToCurrentTimeCheckBox = QtWidgets.QCheckBox()
        #self.scaleYToCurrentTimeCheckBox.setChecked(True)
        #optionsLayout.addWidget(scaleLabel)
        #optionsLayout.addWidget(self.scaleYToCurrentTimeCheckBox)
        #spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        #optionsLayout.addItem(spacer)
        #layout.addLayout(optionsLayout)

