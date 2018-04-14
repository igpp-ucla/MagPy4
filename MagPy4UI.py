
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy

import pyqtgraph as pg

#from qrangeslider import QRangeSlider
#can add in translation stuff later

class UI_MagPy4(object):
    def setupUI(self, MagPy4):

        # gives default window options in top right
        MagPy4.setWindowFlags(QtCore.Qt.Window)
        MagPy4.resize(1400,900)

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

        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionPlot)
        self.toolBar.addAction(self.actionShowData)

        self.glw = pg.GraphicsLayoutWidget()#border=(100,100,100))
        #self.gv = pg.GraphicsView()
        #self.glw = pg.GraphicsLayout()
        #self.gv.setCentralItem(self.glw)

        self.glw.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        mainHoriz = QtWidgets.QHBoxLayout()
        #mainHoriz.addWidget(self.gv)
        mainHoriz.addWidget(self.glw)
        
        layout = QtWidgets.QVBoxLayout(self.centralWidget)

        layout.addLayout(mainHoriz)

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

        optionsLayout = QtWidgets.QHBoxLayout()
        scaleLabel = QtWidgets.QLabel()
        scaleLabel.setText("Scale Y range to current time selection")
        self.scaleYToCurrentTimeCheckBox = QtWidgets.QCheckBox()
        self.scaleYToCurrentTimeCheckBox.setChecked(True)
        optionsLayout.addWidget(scaleLabel)
        optionsLayout.addWidget(self.scaleYToCurrentTimeCheckBox)
        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        optionsLayout.addItem(spacer)

        layout.addLayout(optionsLayout)

 
class UI_PlotTracer(object):
    def setupUI(self, Frame):
        Frame.resize(100,100)

        checkBoxStyle = """
            QCheckBox{spacing: 0px;}
            QCheckBox::indicator{width:32px;height:32px}
            QCheckBox::indicator:unchecked {            image: url(images/checkbox_unchecked.png);}
            QCheckBox::indicator:unchecked:hover {      image: url(images/checkbox_unchecked_hover.png);}
            QCheckBox::indicator:unchecked:pressed {    image: url(images/checkbox_unchecked_pressed.png);}
            QCheckBox::indicator:checked {              image: url(images/checkbox_checked.png);}
            QCheckBox::indicator:checked:hover {        image: url(images/checkbox_checked_hover.png);}
            QCheckBox::indicator:checked:pressed {      image: url(images/checkbox_checked_pressed.png);}
            QCheckBox::indicator:indeterminate:hover {  image: url(images/checkbox_checked_hover.png);}
            QCheckBox::indicator:indeterminate:pressed {image: url(images/checkbox_checked_pressed.png);}
        """

        Frame.setStyleSheet(checkBoxStyle)

        layout = QtWidgets.QVBoxLayout(Frame)


        #drawStyleLayout = QtWidgets.QHBoxLayout()
        #drawStyleComboLabel = QtWidgets.QLabel()
        #drawStyleComboLabel.setText("Draw Style")
        #self.drawStyleCombo = QtWidgets.QComboBox() # add this to PlotTracer instead
        #self.drawStyleCombo.addItem('dots')
        #self.drawStyleCombo.addItem('lines')
        #drawStyleLayout.addWidget(drawStyleComboLabel)
        #drawStyleLayout.addWidget(self.drawStyleCombo)
        #self.drawStyleCombo.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        #drawStyleComboLabel.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        #buttonLayout.addLayout(drawStyleLayout)

        self.clearButton = QtWidgets.QPushButton('Clear')
        self.removePlotButton = QtWidgets.QPushButton('Remove Plot')
        self.addPlotButton = QtWidgets.QPushButton('Add Plot')
        self.plotButton = QtWidgets.QPushButton('Plot')

        self.clearButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.removePlotButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.addPlotButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.plotButton.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.removePlotButton)
        buttonLayout.addWidget(self.addPlotButton)

        spacer = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        buttonLayout.addItem(spacer)
        layout.addLayout(buttonLayout)
        layout.addWidget(self.plotButton)

        self.gridFrame = QtWidgets.QGroupBox('Plot Matrix')
        self.grid = QtWidgets.QGridLayout(self.gridFrame)
        layout.addWidget(self.gridFrame)

        self.fgridFrame = QtWidgets.QGroupBox('Y Axis Link Groups')
        self.fgrid = QtWidgets.QGridLayout(self.fgridFrame)
        layout.addWidget(self.fgridFrame)

        # make invisible stretch to take up rest of space
        layout.addStretch()